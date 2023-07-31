import torch

from nerfdiff.model.tiny_nerf_model import TinyNeRF_model

class TinyNeRF:
    def __init__(self, device):
        self.F_c = TinyNeRF_model().to(device)
        self.chunk_size = 16384
        self.t_n = t_n = 1.0
        self.t_f = t_f = 4.0
        self.N_c = N_c = 32
        self.t_i_c_gap = t_i_c_gap = (t_f - t_n) / N_c
        self.t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

    def get_coarse_query_points(self, ds, os):
        u_is_c = torch.rand(*list(ds.shape[:2]) + [self.N_c]).to(ds)
        t_is_c = self.t_i_c_bin_edges + u_is_c * self.t_i_c_gap
        r_ts_c = os[..., None, :] + t_is_c[..., :, None] * ds[..., None, :]
        return (r_ts_c, t_is_c)

    def render_radiance_volume(self, r_ts, ds, F, t_is):
        r_ts_flat = r_ts.reshape((-1, 3))
        ds_rep = ds.unsqueeze(2).repeat(1, 1, r_ts.shape[-2], 1)
        ds_flat = ds_rep.reshape((-1, 3))
        c_is = []
        sigma_is = []
        for chunk_start in range(0, r_ts_flat.shape[0], self.chunk_size):
            r_ts_batch = r_ts_flat[chunk_start : chunk_start + self.chunk_size]
            ds_batch = ds_flat[chunk_start : chunk_start + self.chunk_size]
            preds = F(r_ts_batch, ds_batch)
            c_is.append(preds["c_is"])
            sigma_is.append(preds["sigma_is"])

        c_is = torch.cat(c_is).reshape(r_ts.shape)
        sigma_is = torch.cat(sigma_is).reshape(r_ts.shape[:-1])

        delta_is = t_is[..., 1:] - t_is[..., :-1]
        one_e_10 = torch.Tensor([1e10]).expand(delta_is[..., :1].shape)
        delta_is = torch.cat([delta_is, one_e_10.to(delta_is)], dim=-1)
        delta_is = delta_is * ds.norm(dim=-1).unsqueeze(-1)

        alpha_is = 1.0 - torch.exp(-sigma_is * delta_is)

        T_is = torch.cumprod(1.0 - alpha_is + 1e-10, -1)
        T_is = torch.roll(T_is, 1, -1)
        T_is[..., 0] = 1.0

        w_is = T_is * alpha_is

        C_rs = (w_is[..., None] * c_is).sum(dim=-2)

        return C_rs

    def __call__(self, ds, os):
        (r_ts_c, t_is_c) = self.get_coarse_query_points(ds, os)
        C_rs_c = self.render_radiance_volume(r_ts_c, ds, self.F_c, t_is_c)
        return C_rs_c