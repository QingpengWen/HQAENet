import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--need_task_scheduling', type=bool, default=False,
                        help='use the task scheduling module.')
    parser.add_argument('--need_data_aligned', type=bool, default=True,
                        help='False --> unaligned data ; True --> aligned data.')
    parser.add_argument('--need_model_aligned', type=bool, default=True,
                        help='False --> complete modality setting ; True --> incomplete modality setting.')
    parser.add_argument('--is_tune', type=bool, default=False,
                        help='tune parameters ?')
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--modelName', type=str, default='gqa_net',
                        help='support hgatt_net/gqa_net')
    parser.add_argument('--datasetName', type=str, default='mosei',
                        help='support mosi/mosei/sims')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='save_model/models/GQA/aligned/MOSI',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='output/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    parser.add_argument('--missing_rates', type=float, nargs='+', default=0)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--def_epoch', type=int, help='running epoch', default=500)
    parser.add_argument('--early_stop', type=int, help='early stop epoch', default=50)
    parser.add_argument('--do_evaluation', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=11111, help='start seed')
    parser.add_argument('--num_seeds', type=int, default=None, help='number of total seeds')
    parser.add_argument('--num_heads', type=int, default=8, help='number of Attention heads')
    parser.add_argument('--num_groups', type=int, default=4, help='number of GQA groups')
    parser.add_argument('--num_layer', type=int, default=5, help='layer of HGAtt')
    parser.add_argument('--exp_name', type=str, default='', help='experiment name')
    parser.add_argument('--diff_missing', type=float, nargs='+', default=None, help='different missing rates for text, audio, and video')
    parser.add_argument('--KeyEval', type=str, default='Loss', help='the evaluation metric used to select best model')
    parser.add_argument('--save_model', action='store_true', help='save the best model in each run (i.e., each seed)')
    parser.add_argument('--use_normalized_data', action='store_true', help='use normalized audio & video data (for now, only for sims)')
    return parser.parse_args()
