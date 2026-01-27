from typing import List, Tuple


def get_fid_is(dir_raw: str, dir_recon: str, feature_extractor_path: str) -> Tuple[float, float]:
    import torch_fidelity
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dir_recon,
        input2=dir_raw,
        samples_shuffle=True,
        samples_find_deep=False,
        samples_find_ext='png,jpg,jpeg',
        samples_ext_lossy='jpg,jpeg',

        cuda=True,
        batch_size=1536,
        isc=True,
        fid=True,

        kid=False,
        kid_subsets=100,
        kid_subset_size=1000,

        ppl=False,
        prc=False,
        ppl_epsilon=1e-4 or 1e-2,
        ppl_sample_similarity_resize=64,
        feature_extractor='inception-v3-compat',
        feature_layer_isc='logits_unbiased',
        feature_layer_fid='2048',
        feature_layer_kid='2048',
        feature_extractor_weights_path=feature_extractor_path,
        verbose=True,

        save_cpu_ram=False,  # using num_workers=0 for any dataset input1 input2
        rng_seed=0,  # FID isn't sensitive to this
    )
    fid = metrics_dict['frechet_inception_distance']
    isc = metrics_dict['inception_score_mean']
    return fid, isc

dir_raw = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/evaluator_gen/imagenet_val_5w_256x256'
dir_recon = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/VVQ/samples_recon_sdvae_val_v1_f16-mar/imagenet-size-256-size-256-seed-0'
feature_extractor_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/changyifan/data/evaluator_gen/weights-inception-2015-12-05-6726825d.pth'

rfid, isc = get_fid_is(dir_raw, dir_recon, feature_extractor_path)

print(rfid, isc)