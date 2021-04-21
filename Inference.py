import torch
import os
import sys

from models import utils
from models import model as model_hub
from torchvision.transforms import transforms
from models.model_cud import CUD_Loss


class Inferencer:

    def __init__(self, args):
        sys.path.insert(0, './model')

        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = self.__init_model()
        checkpoint = torch.load(args.model_path)
        self.model.load_state_dict(checkpoint)
        self.criterion = CUD_Loss(ssim_window_size=5)

        self.trans = transforms.Compose([transforms.ToTensor()
                                         ])

        self.model.eval()

    def start_inference(self, src_path, src_path_y=None):
        path, fn = os.path.split(src_path)
        fn, ext = os.path.splitext(fn)

        input_rgb, input_rgb_deu, input_rgb_diff = utils.load_image_deu_stack(src_path)

        input_rgb = self.trans(input_rgb)
        input_rgb_deu = self.trans(input_rgb_deu)
        input_rgb_diff = self.trans(input_rgb_diff)

        input_src = torch.cat((input_rgb, input_rgb_deu, input_rgb_diff))

        if src_path_y is not None:
            target_rgb = utils.load_image(src_path_y, self.args.grey_scale)
            target_src = self.trans(target_rgb)
            target_src = target_src.unsqueeze(0)

        input_src = input_src.unsqueeze(0)

        output = self.model(input_src, fn=fn)
        output = torch.clamp(output, 0.0, 1.0)
        output = output.squeeze(0)

        if self.args.save_figures:
            utils.save_histogram(input_src, output, target_src, channel='RGB', data_name=fn)

        output = utils.tensor_to_pil(output)
        output = utils.m_invert(output)

        return output
    
    def __init_model(self):

        model = torch.nn.DataParallel(
            model_hub.CUD_NET(num_points=self.args.points,
                              save_figures=self.args.save_figures,
                              clip_threshold=self.args.clip_threshold).to(self.device))
        
        return model
