import os
import configargparse


class ConfigParser(configargparse.ArgParser):
    def __init__(self):
        super().__init__(default_config_files=[
            os.path.join(os.path.dirname(__file__),
                         'livingroom1.yaml')
        ], conflict_handler='resolve')

        # for 4D point cloud dataset
        self.add_argument('--path_datasets', type=str)
        self.add_argument('--path_results', type=str)
        self.add_argument('--dataset_name', type=str)
        self.add_argument('--color_ext', type=str)
        self.add_argument('--nframes', type=int)
        self.add_argument('--bandwidth', type=float)
        self.add_argument('--deci', type=float)
        self.add_argument('--machine', type=str)
        self.add_argument('--zfill', type=int)
        self.add_argument('--l_thres', type=float)
        self.add_argument('--bw_list', nargs='+', type=float)

    def get_config(self):
        config = self.parse_args()
        return config