import torch
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_latent  = nn.Linear(hidden_dim, latent_dim)

        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        latent     = self.FC_latent(h_)

        
        return latent

    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = self.FC_output(h)
        return x_hat
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
                
    def forward(self, x):
        latent = self.Encoder(x)
        x_hat = self.Decoder(latent)
        
        return x_hat, latent
    
    

def load_models():
    x_dim = 20
    hidden_dim = 18
    latent_dim = 15
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim =x_dim)
    movie_model = Model(Encoder=encoder, Decoder=decoder)
    movie_model.load_state_dict(torch.load('movie_vae.pt'))
    movie_model.eval()
    
    x_dim = 3
    hidden_dim = 10
    latent_dim = 15
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim =x_dim)
    user_model = Model(Encoder=encoder, Decoder=decoder)
    user_model.load_state_dict(torch.load('user_vae.pt'))
    user_model.eval()
    
    return movie_model, user_model