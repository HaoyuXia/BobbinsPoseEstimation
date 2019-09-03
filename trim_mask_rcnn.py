import os
import torch
import argparse

def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r

def main():
    parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
    parser.add_argument(
            "--pretrained_path",
            default="/home/xia/e2e_mask_rcnn_R_101_FPN_1x.pth",
            help="path to detectron pretrained weight(.pkl)",
            type=str,
            )
    parser.add_argument(
            "--save_path",
            default="/home/xia/e2e_mask_rcnn_R_101_FPN_1x_trimmed.pth",
            help="path to save the converted model",
            type=str,
            )
    args = parser.parse_args()
    
    DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
    print('detectron path: {}'.format(DETECTRON_PATH))
    
    mrcnn = torch.load(DETECTRON_PATH)
    new_mrcnn = mrcnn
    del mrcnn['optimizer']
    del mrcnn['iteration']
    del mrcnn['scheduler']
    new_mrcnn['model'] = removekey(mrcnn['model'], 
                                   ['module.roi_heads.box.predictor.cls_score.weight',
                                    'module.roi_heads.box.predictor.cls_score.bias',
                                    'module.roi_heads.box.predictor.bbox_pred.weight',
                                    'module.roi_heads.box.predictor.bbox_pred.bias',
                                    'module.roi_heads.mask.predictor.mask_fcn_logits.weight',
                                    'module.roi_heads.mask.predictor.mask_fcn_logits.bias'])
    
    torch.save(new_mrcnn, args.save_path)
    print('saved to {}.'.format(args.save_path))

if __name__ == '__main__':
    main()
    