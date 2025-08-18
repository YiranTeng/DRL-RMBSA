from gym.envs.registration import register

register(
    id="MBEON_Env-v0",
    entry_point="env.mbeon_env:MBEON",
)

register(
    id="DRL-RMBSA-v0",
    entry_point="env.DRL_RMBSA:DRL_RMBSA",
)

register(
    id="DRL-RMBSA-PCA-v0",
    entry_point="env.DRL_RMBSA_PCA:DRL_RMBSA_PCA",
)

register(
    id="KSP-FB-FF-v0",
    entry_point="env.KSP_FB_FF:KSP_FB_FF",
)

register(
    id="KSP-MinMaxF-v0",
    entry_point="env.KSP_MinMaxF:KSP_MinMaxF",
)

register(
    id="KSP-HCP-HMF-v0",
    entry_point="env.KSP_HCP_HMF:KSP_HCP_HMF",
)
