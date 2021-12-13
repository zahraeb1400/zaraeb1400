import torch
import yaml
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
import cv2
import dlib

def load_checkpoints(config_path, checkpoint_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'], **config['model_params']['common_params'])
    if (torch.cuda.is_available()):
        generator.cuda()
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'], **config['model_params']['common_params'])
    if (torch.cuda.is_available()):
        kp_detector.cuda()
    if not(torch.cuda.is_available()):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    if (torch.cuda.is_available()):
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()
    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector, relative=False, adapt_movement_scale=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if (torch.cuda.is_available()):
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if (torch.cuda.is_available()):
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def main_video(input_image,output_video,mode='smile'):
    detector = dlib.get_frontal_face_detector()
    source_image = imageio.imread(input_image)
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for d in dets:
        x = d.left()
        y = d.top()
        w = d.right()
        h = d.bottom()
    source_image = source_image[y - int(h / 3):h + int(h / 10), x - int(w / 5):w + int(w / 5)]
    modes = ['smile', 'suprize', 'shame', 'sneer', 'angry']
    if not mode in modes:
        raise Exception('mode: ' + mode + ' not found!')
    b = modes.index(mode) + 1

    if b == 1:
        reader = imageio.get_reader("./emotions/smile.mp4")
    elif b == 2:
        reader = imageio.get_reader("./emotions/suprize.mp4")
    elif b == 3:
        reader = imageio.get_reader("./emotions/shame.mp4")
    elif b == 4:
        reader = imageio.get_reader("./emotions/sneer.mp4")
    elif b == 5:
        reader = imageio.get_reader("./emotions/angry.mp4")

    fps = reader.get_meta_data()['fps']
    driving_video1 = []
    try:
        for im in reader:
            driving_video1.append(im)
    except RuntimeError:
        pass
    reader.close()
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video1 = [resize(frame, (256, 256))[..., :3] for frame in driving_video1]
    generator, kp_detector = load_checkpoints(config_path="./config/vox-256.yaml", checkpoint_path="./models/vox-cpk.pth.tar")
    predictions = make_animation(source_image, driving_video1, generator, kp_detector, relative=True, adapt_movement_scale=True)
    imageio.mimsave(output_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


main_video('1.jpeg',"result5.mp4",mode='angry')