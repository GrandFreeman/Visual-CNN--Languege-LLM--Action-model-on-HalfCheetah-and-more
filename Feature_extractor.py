# ---------------------------
# 3) Custom feature extractor:
#    CNN(image) + Embedding(text)
# ---------------------------

class ImageTextExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256, vocab_size: int = len(VOCAB)):
        super().__init__(observation_space, features_dim)

        # image: (B, 3, 84, 84)
        n_channels = observation_space.spaces["image"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, *observation_space.spaces["image"].shape)
            cnn_out_dim = self.cnn(sample).shape[1]

        # Get the actual (stacked) state dimension from the observation space
        actual_stacked_state_dim = observation_space.spaces["state"].shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(actual_stacked_state_dim, 32),
            nn.ReLU()
        )


        self.text_emb = nn.Embedding(vocab_size, 32, padding_idx=VOCAB[PAD_TOKEN])
        self.text_proj = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, observations):
        # SB3 will feed tensors here.
        img = observations["image"].float()
        if img.max() > 1.5:
            img = img / 255.0

        text = observations["text"].long()  # (B, L)
        mask = (text != VOCAB[PAD_TOKEN]).float().unsqueeze(-1)  # (B, L, 1)

        txt_emb = self.text_emb(text)  # (B, L, 32)
        txt_feat = (txt_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        txt_feat = self.text_proj(txt_feat)

        # Process the state observation
        state = observations["state"]
        state_feat = self.state_mlp(state) # Process state using its MLP

        img_feat = self.cnn(img)
        fused = th.cat([img_feat, txt_feat, state_feat], dim=1)
        return self.fusion(fused)
