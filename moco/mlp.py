from torch import nn
import torch.nn.functional as F

    
class Encoder(nn.Module): 
    def __init__(self,
                 in_features,
                 num_cluster,
                 latent_features = [1024, 512, 128],
                 device="cpu",
                 p=0.0):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.device = device

        layers = []
        layers.append(nn.Dropout(p=p))
        for i in range(len(latent_features)):
            if i == 0:
                layers.append(nn.Linear(in_features, latent_features[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(latent_features[i-1], latent_features[i]))
                layers.append(nn.ReLU())
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers) 


        self.fc = nn.Linear(latent_features[-1], num_cluster)
        self.classifier = nn.Sequential(nn.Linear(latent_features[-1],int(latent_features[-1]/2)),
                                        nn.BatchNorm1d(int(latent_features[-1]/2)),
                                    nn.ReLU(),
                                    nn.Linear(int(latent_features[-1]/2),num_cluster)
                                    )
    def forward(self, x):
        h = self.encoder(x)
        # h = self.encoder_2(h)
        out = self.fc(h)
        out = self.classifier(out)

        return out
    
    def get_embedding1(self, x):
        latent = self.encoder(x)
        # latent = self.encoder_2(latent)

        return latent
    
    def get_embedding2(self, x):
        latent = self.encoder(x)
        latent = self.fc(latent)
        return latent
    
    def get_embedding3(self, x):
        latent = self.encoder(x)
        latent = self.fc(latent)
        latent = self.classifier(latent)
        return latent