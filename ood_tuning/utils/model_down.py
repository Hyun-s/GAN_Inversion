import gdown
import argparse
import os
downs = {
    'BDInvert_base_encoder':{        
        "url" : "https://drive.google.com/file/d/1nMhKTuSbe9rjL2j_9rhTDCEQYHC7mmRc/view?usp=sharing",
        "output" : "BDInvert_encoder_stylegan2_ffhq1024_basesize16.pth"
    },
    
    'style_gan':{
        "url": "https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",
        "output": "stylegan2-ffhq-config-f.pt"
    },
    'style_gan_module':{
        "url":'https://drive.google.com/uc?id=1pqeMD8fKswJFW2aWi1ObZvEaIz5OKQ9w',
        'output':'stylegan2_ffhq1024.pth'
    },
    'arcface':{
        "url":'https://drive.google.com/file/d/12VHLJhEbUIYOZdoWwt3IU6wE69ulVW9W/view?usp=sharing',
        'output':'model_ir_se50.pth'
    },
    'sample_image':{
        "url":'https://drive.google.com/file/d/18f_5DfJCsxX4olYEXTZsWZYKFKk8yg3U/view?usp=sharing',
        "output": '66998.png'
    },
    'metface_processed':{
        "url":'https://drive.google.com/file/d/1AA-k_x_wu4VO4KsTphSvxujSRoqGHCP3/view?usp=sharing',
        "output":'processed_metface.zip'
    },
    'data':{
        "url":'https://drive.google.com/file/d/1f6Fi4rrhAtw8Gq6Fvvw7kmzvkvXdSM-R/view?usp=sharing',
        "output":'BDInvert_1200.zip'
    }
}


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Pre-trained model downloads.')

    parser.add_argument('--down_path', type=str, default=None,
                        help='Path to the download pre-trained model weights.')

    return parser.parse_args()


def main():
    args = parse_args()
    BDInvert_e = downs['BDInvert_base_encoder']
    style_gan = downs['style_gan_module']
    metface_p = downs['metface_processed']
    data = downs['data']
    arcface = downs['arcface']

    gdown.download(style_gan['url'], os.path.join(args.down_path,
                                                  style_gan['output']), quiet=False)
    gdown.download(arcface['url'], os.path.join(args.down_path,
                                                  arcface['output']), quiet=False, fuzzy=True)
    gdown.download(BDInvert_e['url'], os.path.join(args.down_path,
                                                  BDInvert_e['output']), quiet=False, fuzzy=True)
    gdown.download(metface_p['url'], os.path.join(args.down_path,
                                                  metface_p['output']), quiet=False, fuzzy=True)
    gdown.download(data['url'], os.path.join(args.down_path,
                                                  data['output']), quiet=False, fuzzy=True)