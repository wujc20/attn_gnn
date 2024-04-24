import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer,DSAttention
from layers.Embed import DataEmbedding,FixedAbsolutePositionEmbedding,DataEmbedding_rope,DataEmbedding_wo_pos
from einops import rearrange, repeat


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


def FFT_for_Period(x, k=4):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1) 
    frequency_list[0] = float('-inf')
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period=[1]
    for top in top_list:
        #print(x.shape[1],top,x.shape[1] / top)
        period = np.concatenate((period,[math.ceil(x.shape[1] / top)])) #    
    return period, abs(xf).mean(-1)[:, top_list] #
class moving_avg(nn.Module):
    """
    Moving average block
    """
    def __init__(self, kernel_size):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size, padding=0) 

    def forward(self, x):
        # batch seq_len channel
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x #batch seq_len channel

class multi_scale_data(nn.Module):
    '''
    Concantenate Different Scales
    '''
    def __init__(self, kernel_size,return_len):
        super(multi_scale_data, self).__init__()
        self.kernel_size = kernel_size
        self.max_len = return_len
        self.moving_avg = [moving_avg(kernel) for kernel in kernel_size]
    def forward(self, x):
        # batch seq_len channel
        different_scale_x = []
        for func in self.moving_avg:
            moving_avg = func(x)
            different_scale_x.append(moving_avg)
            #print(moving_avg.shape)
        multi_scale_x=torch.cat(different_scale_x,dim=1)
        # ensure fixed shape: [batch, max_len, variables]
        if multi_scale_x.shape[1]<self.max_len: #padding
            padding = torch.zeros([x.shape[0], (self.max_len - (multi_scale_x.shape[1])), x.shape[2]]).to(x.device)
            multi_scale_x = torch.cat([multi_scale_x,padding],dim=1)
        elif multi_scale_x.shape[1]>self.max_len: #trunc
            multi_scale_x = multi_scale_x [:,:self.max_len,:]
        #print(multi_scale_x.shape)
        return multi_scale_x

class nconv(nn.Module):
    def __init__(self,gnn_type):
        super(nconv,self).__init__()
        self.gnn_type = gnn_type
    def forward(self,x, A):
        if self.gnn_type =='time':
            x = torch.einsum('btdc,tw->bwdc',(x,A))
        else:
            x = torch.einsum('btdc,dw->btwc',(x,A))
        return x.contiguous()
    
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,gnn_type,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv(gnn_type)
        self.gnn_type=gnn_type
        self.c_in = (order+1)*c_in 
        self.mlp = nn.Linear(self.c_in,c_out)
        self.dropout = dropout
        self.order = order
        self.act = nn.GELU()
    def forward(self,x,a):
        # in: b t dim d_model
        # out: b t dim d_model
        out = [x]
        x1 = self.nconv(x,a)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1,a)
            out.append(x2)
            x1 = x2
        h=torch.cat(out,dim=-1)
        h=self.mlp(h)
        h=self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h#.unsqueeze(1)
class single_scale_gnn(nn.Module):
    def __init__(self, configs):
        super(single_scale_gnn, self).__init__()

        self.tk=configs.tk
        self.scale_number=configs.scale_number + 6
        self.use_tgcn=configs.use_tgcn
        self.use_ngcn=configs.use_ngcn
        self.init_seq_len = configs.seq_len 
        self.pred_len = configs.pred_len
        self.ln = nn.ModuleList()

        #modifired
        #self.channels = configs.enc_in
        self.channels = configs.d_model

        self.individual = configs.individual
        self.dropout=configs.dropout
        self.device='cuda:'+str(configs.gpu)
        self.GraphforPre = False
        self.tvechidden = configs.tvechidden
        self.tanh=nn.Tanh()

        #modified
        self.d_model = configs.hidden
        # self.d_model = 32
        
        self.start_linear = nn.Linear(1,self.d_model)

        #here modified
        self.seq_len = self.init_seq_len+self.init_seq_len # max_len (i.e., multi-scale shape)
        
        # self.seq_len = self.init_seq_len

        self.timevec1 = nn.Parameter(torch.randn(self.seq_len, configs.tvechidden).to(self.device), requires_grad=True).to(self.device) 
        self.timevec2 = nn.Parameter(torch.randn(configs.tvechidden, self.seq_len).to(self.device), requires_grad=True).to(self.device)
        self.tgcn = gcn(self.d_model,self.d_model,self.dropout,gnn_type='time')
        self.nodevec1 = nn.Parameter(torch.randn(self.channels,  configs.nvechidden).to(self.device), requires_grad=True).to(self.device)
        self.nodevec2 = nn.Parameter(torch.randn( configs.nvechidden, self.channels).to(self.device), requires_grad=True).to(self.device)
        self.gconv = gcn(self.d_model,self.d_model,self.dropout,gnn_type='nodes')
        self.layer_norm = nn.LayerNorm(self.channels) 
        self.grang_emb_len = math.ceil(self.d_model//4)
        self.graph_mlp = nn.Linear(2*self.tvechidden,self.grang_emb_len)
        self.act = nn.Tanh()
        if self.use_tgcn:
            dim_seq = 2*self.d_model
            if self.GraphforPre:
                dim_seq = 2*self.d_model+self.grang_emb_len#2*self.seq_len+self.grang_emb_len
        else:
            dim_seq = 2*self.seq_len   #2*self.seq_len   
        self.Linear = nn.Linear(dim_seq, 1) # map to intial scale
    def logits_warper_softmax(self,adj,indices_to_remove,filter_value=-float("Inf")):
        adj = F.softmax(adj.masked_fill(indices_to_remove,filter_value),dim=0)
        return adj
    def logits_warper(self,adj,indices_to_remove,mask_pos,mask_neg,filter_value=-float("Inf")):
        #print('adj:',adj)
        mask_pos_inverse = ~mask_pos
        mask_neg_inverse = ~mask_neg
        # Replace values for mask_pos rows
        processed_pos =  mask_pos * F.softmax(adj.masked_fill(mask_pos_inverse,filter_value),dim=-1) 
        # Replace values for mask_neg rows
        processed_neg = -1 * mask_neg * F.softmax((1/(adj+1)).masked_fill(mask_neg_inverse,filter_value),dim=-1) 
        # Combine processed rows for both cases
        processed_adj = processed_pos + processed_neg
        return processed_adj
    def add_adjecent_connect(self,mask):
        s=np.arange(0,self.seq_len-1) # torch.arange(start=0,end=self.seq_len-1)
        e=np.arange(1,self.seq_len)
        forahead = np.stack([s,e],0)
        back = np.stack([e,s],0)
        all = np.concatenate([forahead,back],1)
        mask[all] = False
        return mask
    def add_cross_scale_connect(self,adj,periods):
        max_L = self.seq_len
        mask=torch.tensor([],dtype=bool).to(adj.device)
        k=self.tk
        k = 20
        min_total_corss_scale_neighbors = 5 #  number
        start = 0
        end = 0
        for period in periods:
            ls=self.init_seq_len//period # time node number at this scale
            end=start+ls #
            if end > max_L: # 
                end = max_L #
                ls = max_L-start #+
            kp=k//period 
            kp=max(kp,min_total_corss_scale_neighbors)
            kp=min(kp,ls) # prevent kp exceeding ls
            mask = torch.cat([mask,adj[:,start:end] < torch.topk(adj[:,start:end], k=kp)[0][..., -1, None]],dim=1) 
            start=end
            if start==max_L:
                break  
        if start<max_L:
            mask=torch.cat([mask,torch.zeros(self.seq_len,max_L-start,dtype=bool).to(mask.device)],dim=1)
        return mask
    def add_cross_var_adj(self,adj):
        k = 3
        #here modified
        # print(adj.shape)
        # print(int(np.log(adj.shape[0]) * np.e),int(0.2*adj.shape[0]))
        # if adj.shape[0] < 100:
        k = max(k,int(0.15*adj.shape[0]))
        # k=min(k,adj.shape[0])
        # else:
            # k = max(k , int(np.log(adj.shape[0]) * np.e))
        mask = (adj < torch.topk(adj, k=adj.shape[0]-k)[0][..., -1, None]) * (adj > torch.topk(adj, k=adj.shape[0]-k)[0][..., -1, None])
        mask_pos = adj >= torch.topk(adj, k=k)[0][..., -1, None] 
        mask_neg = adj <= torch.kthvalue(adj, k=k)[0][..., -1, None]
        return mask,mask_pos,mask_neg
    def get_time_adj(self,periods):
        adj=F.relu(torch.einsum('td,dm->tm',self.timevec1,self.timevec2))
        mask = self.add_cross_scale_connect(adj,periods)
        mask = self.add_adjecent_connect(mask)
        adj = self.logits_warper_softmax(adj=adj,indices_to_remove=mask)
        return adj
    def get_var_adj(self):
        adj=F.relu(torch.einsum('td,dm->tm',self.nodevec1,self.nodevec2))
        mask,mask_pos,mask_neg=self.add_cross_var_adj(adj)
        adj = self.logits_warper(adj,mask,mask_pos,mask_neg)
        return adj
    def get_time_adj_embedding(self,b):
        graph_embedding = torch.cat([self.timevec1,self.timevec2.transpose(0,1)],dim=1) 
        graph_embedding = self.graph_mlp(graph_embedding)
        graph_embedding = graph_embedding.unsqueeze(0).unsqueeze(2).expand([b,-1,self.channels,-1])
        return graph_embedding
    def expand_channel(self,x):
        # x: batch seq_len dim 
        # out: batch seq dim d_model
        x=x.unsqueeze(-1)
        x=self.start_linear(x)
        return x
    def forward(self, x):
        # x: [Batch, Input length, Dim]
        # print(x.shape)
        periods,_=FFT_for_Period(x,self.scale_number)
        multi_scale_func = multi_scale_data(kernel_size=periods,return_len=self.seq_len)
        x = multi_scale_func(x)  # Batch 2*seq_len channel
        # # print(x.shape)
        x =self.expand_channel(x)
       
        batch_size=x.shape[0]
        x_ = x
        #if else 是modif后的结构，原代码为else部分
        # if self.use_tgcn and self.use_ngcn:
        #     time_adp =  self.get_time_adj(periods)
        #     # print(time_adp.shape)
        #     gcn_adp =  self.get_var_adj()
        #     # print(gcn_adp.shape)
        #     x = self.gconv(x, gcn_adp) + self.tgcn(x,time_adp) + x
        #     self.time_adp = time_adp
        #     self.gcn_adp = gcn_adp
        # else:
        if self.use_tgcn:
            time_adp =  self.get_time_adj(periods)
            # print(time_adp.shape)
            x = self.tgcn(x,time_adp)+x
            self.time_adp = time_adp

        if self.use_ngcn:
            gcn_adp =  self.get_var_adj()
            # print(gcn_adp.shape)
            x = self.gconv(x, gcn_adp)+x
            self.gcn_adp = gcn_adp

        x = torch.cat([x_ , x],dim=-1)
        if self.use_tgcn and self.GraphforPre:
            graph_embedding = self.get_time_adj_embedding(b=batch_size)
            x=torch.cat([x,graph_embedding],dim=-1)
        # print(x.shape)
        x = self.Linear(x).squeeze(-1)
        # print(x.shape)
        x = F.dropout(x,p=self.dropout,training=self.training)
        return x[:,:self.init_seq_len,:] #+ x[:,self.init_seq_len:self.init_seq_len+self.init_seq_len,:] # [Batch, init_seq_len(96), variables]


class AttnMask:
    def __init__(self, attn):
        
        self.mask = attn != 0

class AttnMask_pos:
    def __init__(self, attn):
        
        self.mask = attn == 0


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  ###RMS Norm公式

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
        #origin
        # if rms_norm is not None and x.is_cuda:
        #     return rms_norm(x, self.weight, self.eps)
        # else:
        #     output = self._norm(x.float()).type_as(x)
        #     return output * self.weight


class Model(nn.Module):
    '''
    CrossGNN
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.set_node = 6
        # if self.set_node == 6:
        #     configs.d_model = configs.enc_in


        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.graph_encs = nn.ModuleList()
        self.graph_decs = nn.ModuleList()
        self.dropout = nn.Dropout(configs.dropout)


        # configs.d_model = configs.dec_in
        configs.e_layers = 1
        configs.d_layers = 1
        self.dec_layers = configs.d_layers
        # self.GraphforPre = None
        # self.use_tgcn = None

        
        self.scale_number=configs.scale_number *4
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.norm3 = nn.LayerNorm(configs.seq_len)
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        self.FN = nn.Linear(configs.enc_in, configs.enc_in)

        self.enc_layers = configs.e_layers
        self.anti_ood = configs.anti_ood
      
        for i in range(self.enc_layers):
            self.graph_encs.append(single_scale_gnn(configs=configs))
            configs.seq_len = configs.pred_len + configs.label_len
            self.graph_decs.append(single_scale_gnn(configs=configs))
            configs.seq_len = self.seq_len

        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention


        #Attention 

        self.attention_lay =  AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads)
                
        self.attention_lay_var =  AttentionLayer(
                    DSAttention(True, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.seq_len, configs.n_heads)
            

        self.attention_dec =  AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.pred_len, configs.n_heads)
                

        # Embedding
        self.enc_embedding = FixedAbsolutePositionEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # # Encoder
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
            
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )
        # #Decoder
        
        # self.DtD = nn.Linear(configs.d_model,configs.dec_in)
        
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        #this can't be 
        self.dec_embedding = FixedAbsolutePositionEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
        
        #     # configs.d_model = configs.dec_in

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        self.EtG = nn.Linear(configs.d_model, configs.enc_in)
        # self.GtD = nn.Linear(configs.enc_in, configs.d_model)
        # self.TimetM = nn.Linear((configs.seq_len*2)**2,(configs.label_len+configs.pred_len)*configs.batch_size*4)
        # self.VartM = nn.Linear(configs.enc_in**2,(configs.seq_len+configs.pred_len)*4*4)
        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

        self.rmsnorm = RMSNorm(configs.enc_in)
        self.rmsnorm1 = RMSNorm(configs.d_model)

        self.weight = nn.Parameter(torch.ones(configs.c_out))
    def forward(self,x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # print(x_mark_dec)
        # x: [Batch, Input length, Variables]
        if self.anti_ood:
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last
        
        # enc_out = self.enc_embedding(x, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print(x.shape)
        x_raw = x.clone().detach()

        time_map = []
        var_map = []
        
        # Normalization
        mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / std_enc
        

        # B x S x E, B x 1 x E -> B x 1, positive scalar
        
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        x = self.enc_embedding(x, x_mark_enc)
       
        set_node = self.set_node
        var_map_mask = None
        time_map_mask = None
        # for i in range(self.enc_layers): 
        #     if not var_map_mask:
        #         var_map_mask = var_map[i] 
        #     else:
        #         var_map_mask += var_map[i]
        g_x = x.clone()
        g_x =self.rmsnorm1(g_x)

        for i in range(self.enc_layers):
            # print('ori',g_x)
            # g_x = self.norm3(g_x)
            g_x = self.graph_encs[i](g_x)
            time_map.append(self.graph_encs[i].time_adp)
            var_map.append(self.graph_encs[i].gcn_adp)
            # print(x.shape) 
            # if self.state == 0:
                # print('Through encod-decoder')
            if var_map_mask == None:
                var_map_mask = var_map[i] 
            else:
                var_map_mask =var_map_mask + var_map[i]

            if time_map_mask == None:
                time_map_mask = time_map[i] 
            else:
                time_map_mask =time_map_mask + time_map[i]
            # print('after',g_x)
            # g_x = self.norm1(g_x)
            #set8
            # periods,_=FFT_for_Period(x,self.scale_number)
            # multi_scale_func = multi_scale_data(kernel_size=periods,return_len=self.seq_len*2)
            # # print(x.shape)
            # x = multi_scale_func(x)
            

        # x = x.permute(0,2,1)
        # print(var_map[0][0])
        # var_map_mask = None
        # for i in range(self.enc_layers): 
        #     if not var_map_mask:
        #         var_map_mask = var_map[i] 
        #     else:
        #         var_map_mask += var_map[i]
        if var_map_mask == None:
            mask_map = None
        else:
            # mask_map = AttnMask(var_map_mask)
            mask_map = AttnMask(time_map_mask[:96,:96])
        # print(x.shape,var_map[0].shape,time_map[0].shape)
        # base_x = x
        # new_x, attn = self.attention_lay(
        # base_x, base_x, base_x,
        # attn_mask=mask_map,tau =tau,delta = delta)
        # new_x =self.norm1(self.activation(new_x))
        # new_x = self.dropout(new_x)
        # new_x = new_x.permute(0,2,1)

        mask_map_var = AttnMask(var_map[-1])
        var_X = (x).permute(0,2,1)
        # print('var_x',var_X)
        new_x_var, attn = self.attention_lay_var(
        var_X, var_X, var_X,
        attn_mask=mask_map_var,tau =None,delta = None)   
        # print(mask_map_var.mask)
        # print(new_x,new_x_var,attn)
        # new_x_var =self.norm3(self.activation(new_x_var)).permute(0,2,1)
        new_x_var = new_x_var.permute(0,2,1)
     
        # x = x.permute(0,2,1)
        # x = x[:,:self.seq_len,:]
        x = (x + new_x_var)
        # print(x)
        # x = g_x + new_x
        # x = self.norm2(x)
        # print(x)
        # x = self.FN(x) 
        #set_8
        # pred_x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        #

        # x_mask_map = AttnMask_pos(time_map_mask[:self.seq_len+self.label_len,:144])
        cross_mask_map = AttnMask_pos(time_map_mask[:96,:96])
        # print(cross_mask_map.mask)
        # print(cross_mask_map.mask[0])
        if self.anti_ood:
            dec_last = x_dec[:,self.configs.label_len-1:self.configs.label_len,:].detach()
            # print(dec_last)
            x_dec = x_dec - dec_last
        
        # mean_enc = x_dec.mean(1, keepdim=True).detach()  # B x 1 x E
        # x = x - mean_enc
        # std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        # x = x / std_enc
        

        dec_out = self.dec_embedding(x_dec,x_mark_dec)
        
        # for i in range(self.enc_layers):
            # print(dec_out.shape)
        dec_out =  self.graph_decs[-1](dec_out)
        # dec_out = self.norm2(dec_out)
        # print('prepare',dec_out,x)
        x = self.decoder(dec_out, x, x_mask=None, cross_mask=None, tau=None, delta=None)
        pred_x = x[:, -self.pred_len:, :] 

        pred_x_g = self.EtG(self.Linear(g_x.permute(0,2,1)).permute(0,2,1))
        
       
        # pred_x = self.rmsnorm(pred_x)
        # pred_x = self.s4d(pred_x)[0]
        # print('pred',pred_x)
        
        pred_x = pred_x * std_enc + mean_enc
        pred_x = self.weight *  pred_x +  (1 - self.weight) * pred_x_g
        # print(self.weight)
        # print('pred',pred_x)
        # x = pred_x
        # for i in range(self.dec_layers):
        #     x = x.permute(0,2,1)
        #     mask_map = AttnMask(var_map[0])
        #     # mask_map = None
        #     # print(var_map[0].shape,time_map[0].shape)
        #     new_x, attn = self.attention_dec(
        #     x, x, x,
        #     attn_mask=mask_map)
        #     new_x = self.dropout(new_x)
        #     new_x =self.norm2(self.activation(new_x))
        #     new_x = new_x.permute(0,2,1)

        #     # x = x.permute(0,2,1)
        #     # # x = x[:,:self.seq_len,:]
        #     # x = x - new_x

        # pred_x = new_x
        # print(time_map[0].shape)
        #if set_node == 6:
            # x = x.permute(0,2,1)
            # mask_map = AttnMask(var_map[0])
            # new_x, attn = self.attention_lay(
            # x, x, x,
            # attn_mask=mask_map)
            # x = x + self.dropout(new_x)

            # x = x.permute(0,2,1)
            # pred_x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        #code of set 6:
        # if set_node == 7:
        #     periods,_=FFT_for_Period(x,self.scale_number)
        #     multi_scale_func = multi_scale_data(kernel_size=periods,return_len=self.seq_len*2)
        #     # print(x.shape)
        #     x = multi_scale_func(x)
        #     # print(x.shape)
            
        #     x = x.permute(0,2,1)
        #     mask_map = AttnMask(var_map[0])
        #     # print(var_map[0].shape,time_map[0].shape)
        #     new_x, attn = self.attention_lay(
        #     x, x, x,
        #     attn_mask=mask_map)
        #     x = x + self.dropout(new_x)

        #     x = x.permute(0,2,1)
        #     x = x[:,:self.seq_len,:]
        #     pred_x = self.Linear(x.permute(0,2,1)).permute(0,2,1)


        #code of set 5:
        #no antiood/use_tgcn/GraphforPre and with e/d layer all equal 2
        # if set_node == 5:
        #     x = self.enc_embedding(x, None)
        #     dec_out = self.dec_embedding(x_dec,None)
        #     x = self.decoder(dec_out, x, x_mask=None, cross_mask=None)
        #     pred_x = x[:, -self.pred_len:, :] 
        # print(x_dec.shape)
        #code of set4:
        
        # x += enc_out
        # x = self.GtD(x)
        # print(time_map[0].shape,x_dec.shape,time_map[0].reshape(-1,192,4).shape)
        # mask1 = self.TimetM(time_map[0].flatten()).reshape(-1,x_dec.shape[1],4)
        # mask2 = self.VartM(var_map[0].flatten()).reshape(4,-1,4)
        # print(mask1.shape)   
        # print(x_mark_dec.shape)
        # dec_out = self.dec_embedding(x_dec,None)
        # print(dec_out.shape,x.shape)
        # dec_out = self.DtD(dec_out)
        # print(dec_out.shape)
        # print(x_mark_enc.mask)
        # x = self.decoder(dec_out, x, x_mask=None, cross_mask=None)
        # pred_x = x[:, -self.pred_len:, :] 
        # print(x.shape)
        # pred_x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

        if self.anti_ood:
                pred_x = pred_x  + seq_last
        return pred_x # [Batch, Output length, Variables]
