from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size, dropout,num_layer,output_features,bidirectional=False):
        super(LSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layer = num_layer
        self.output_features = output_features
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            dropout =0 if num_layer == 1 else self.dropout,
                            num_layers=self.num_layer,
                            bias=True,
                            batch_first=True,
                            bidirectional=bidirectional)
        # We use last output value which has size like this for the prediction, since the size of each output in output value is (B,H)
        # The same as ct and ht                    
        self.output = nn.Linear(in_features=self.hidden_size, out_features=self.output_features)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, X):
        # Because output(B,H) of each t, which is the size of ct and ht, so we want the start ct and ht like this
        cell_state = torch.zeros(size=(self.num_layer, X.size(0), self.hidden_size), device=X.device)
        hidden_state = torch.zeros(size=(self.num_layer, X.size(0), self.hidden_size), device=X.device)
        # Return output,(hn,cn), but we just care about the output value
        
        out,_ = self.lstm(X,(hidden_state,cell_state))
        # # We just want to use the last output value
        # v1
        return self.output(out[:,-1,:])

        # v2
        # Lấy bước cuối
        # last_out = out[:, -1, :]
        # Áp dụng Dropout trước khi đưa ra kết quả cuối v2
        # last_out = self.dropout_layer(last_out)
        # return self.output(last_out)
        
        # v3 Tức là 6 đầu ra thì nó co lại 1 đầu thời gian kiểu z
        # out_mean = torch.mean(out, dim=1)
        # return self.output(out_mean)

class LSTMV2(nn.Module):
    def __init__(self,input_size,hidden_size, dropout,num_layers,output_features,bidirectional=False):
        super(LSTMV2,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.output_features = output_features
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            dropout =0 if num_layers == 1 else self.dropout,
                            num_layers=self.num_layers,
                            bias=True,
                            batch_first=True,
                            bidirectional=False)
        # We use last output value which has size like this for the prediction, since the size of each output in output value is (B,H)
        # The same as ct and ht                    
        self.output = nn.Linear(in_features=self.hidden_size, out_features=self.output_features)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, X):
        num_directions = 2 if self.lstm.bidirectional else 1
        # Because output(B,H) of each t, which is the size of ct and ht, so we want the start ct and ht like this
        cell_state = torch.zeros(size=(self.num_layers * num_directions, X.size(0), self.hidden_size), device=X.device)
        hidden_state = torch.zeros(size=(self.num_layers * num_directions, X.size(0), self.hidden_size), device=X.device)
        # Return output,(hn,cn), but we just care about the output value
        
        out,_ = self.lstm(X,(hidden_state,cell_state))
        # # We just want to use the last output value
        # v1 chạy lại với chỉnh lr luôn
        return self.output(out[:,-1,:])

        # v2
        # Lấy bước cuối
        # last_out = out[:, -1, :]
        # # Áp dụng Dropout trước khi đưa ra kết quả cuối v2
        # last_out = self.dropout_layer(last_out)
        # return self.output(last_out)
        
        # v3 Tức là 6 đầu ra thì nó co lại 1 đầu thời gian kiểu z
        # out_mean = torch.mean(out, dim=1)
        # return self.output(out_mean)

        # Bước quan trọng: Mean Pooling dọc theo dim 1 (thời gian)
        # out = out.mean(dim=1) 
        
        # # Áp dụng Dropout để ổn định các đặc trưng
        # out = self.dropout_layer(out)
        
        # return self.output(out)

# Regression
class LSTMV3(nn.Module):
    def __init__(self,input_size,hidden_size, dropout,num_layers,output_features,bidirectional=False):
        super(LSTMV3,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.output_features = output_features
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            dropout =0 if num_layers == 1 else self.dropout,
                            num_layers=self.num_layers,
                            bias=True,
                            batch_first=True,
                            bidirectional=bidirectional)
        # We use last output value which has size like this for the prediction, since the size of each output in output value is (B,H)
        # The same as ct and ht                    
        self.output = nn.Linear(in_features=self.hidden_size, out_features=self.output_features)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, X):
        # Because output(B,H) of each t, which is the size of ct and ht, so we want the start ct and ht like this
        cell_state = torch.zeros(size=(self.num_layers, X.size(0), self.hidden_size), device=X.device)
        hidden_state = torch.zeros(size=(self.num_layers, X.size(0), self.hidden_size), device=X.device)
        # Return output,(hn,cn), but we just care about the output value
        
        out,_ = self.lstm(X,(hidden_state,cell_state))
        # # We just want to use the last output value
        # v1
        return self.output(out[:,-1,:])

        # v2
        # Lấy bước cuối
        # last_out = out[:, -1, :]
        # # Áp dụng Dropout trước khi đưa ra kết quả cuối v2
        # last_out = self.dropout_layer(last_out)
        # return self.output(last_out)
        
        # v3 Tức là 6 đầu ra thì nó co lại 1 đầu thời gian kiểu z
        # out_mean = torch.mean(out, dim=1)
        # return self.output(out_mean)

        # Bước quan trọng: Mean Pooling dọc theo dim 1 (thời gian)
        # out = out.mean(dim=1) 
        
        # # Áp dụng Dropout để ổn định các đặc trưng
        # out = self.dropout_layer(out)
        
        # return self.output(out)