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
                            dropout =self.dropout,
                            num_layers=self.num_layer,
                            bias=True,
                            batch_first=True,
                            bidirectional=bidirectional)
        # We use last output value which has size like this for the prediction, since the size of each output in output value is (B,H)
        # The same as ct and ht                    
        self.output = nn.Linear(in_features=self.hidden_size, out_features=self.output_features)
    
    def forward(self, X):
        # Because output(B,H) of each t, which is the size of ct and ht, so we want the start ct and ht like this
        cell_state = torch.zeros(size=(self.num_layer, X.size(0), self.hidden_size), device=X.device)
        hidden_state = torch.zeros(size=(self.num_layer, X.size(0), self.hidden_size), device=X.device)
        # Return output,(hn,cn), but we just care about the output value
        out,_ = self.lstm(X,(hidden_state,cell_state))
        # We just want to use the last output value
        return self.output(out[:,-1,:])


class LSTMWithATT(nn.Module):
    def __init__(self,input_size,hidden_size, dropout,num_layer,output_features):
        super(LSTMWithATT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layer = num_layer
        self.output_features = output_features
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            dropout =self.dropout,
                            num_layers=self.num_layer,
                            bias=True,
                            batch_first=True,
                            bidirectional=False)
        self.linearLayer = nn.Linear(in_features= self.hidden_size, out_features= self.output_features)
        # Để simplify thì ở đây mình sẽ chỉ xét query là hidden_state cuối
        # key và value sẽ là các timestamp ở mỗi đầu ra của LSTM
        # Mục tiêu là tính xem những output đặc trưng nào ảnh hưởng đến kết quả ở đầu ra nhất
        # Lấy cái đấy làm đặc trưng
        # https://github.com/siddharth17196/LSTM-with-attention/blob/master/models.py
    def attention_net(self, output_layer, final_hidden_size):
        # final_hidden_size shape [1, B, H]
        # output layer shape [B, T, H]
        finalH = final_hidden_size[-1] # [B, H]
        finalH = finalH.unsqueeze(2) # [B, H, 1]
        # Đưa về dạng này nhằm: khi nhân 2 ma trận finalH và output_layer
        # Theo quy tắc nhân ma trận sẽ nhân từ 2 ma trận trong cùng là [H, 1] và [T, H] trước 
        # H sẽ được nhân qua lần lượt các timestamp khác nhau từ đó tính được attention weight
        attWeights = torch.bmm(output_layer, finalH) # [B, T, 1]
        # softmax
        # Sau khi có weight nhân với các timestamp ở values để biết mức độ attt của từng timestamps
        # lên timestamp cuối cùng
        soft_attn_weights = torch.softmax(attWeights, 1) # [B, T, 1]
        new_hidden_state = torch.bmm(torch.permute(soft_attn_weights, dims= [0, 2, 1]), output_layer) # (B, 1, H)
        return new_hidden_state.squeeze(1) # (B, H)
    
    def forward(self, X):
        hiddenState = torch.zeros(size=(self.num_layer, X.size(0), self.hidden_size), device= X.device)
        cellState = torch.zeros(size=(self.num_layer, X.size(0), self.hidden_size), device= X.device)
        
        outputValue,(finalHState, finalCState) = self.lstm(X, (hiddenState, cellState))
        new_hidden_state = self.attention_net(outputValue, finalHState)

        result = self.linearLayer(new_hidden_state)
        return result