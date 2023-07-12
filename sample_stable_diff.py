import argparse
import json
import os
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline

torch.backends.cuda.matmul.allow_tf32 = True

mit_classes = {'airport_inside': 0, 'artstudio': 1, 'auditorium': 2, 'bakery': 3, 'bar': 4, 
              'bathroom': 5, 'bedroom': 6, 'bookstore': 7, 'bowling': 8, 'buffet': 9, 
              'casino': 10, 'children_room': 11, 'church_inside': 12, 'classroom': 13, 
              'cloister': 14, 'closet': 15, 'clothingstore': 16, 'computerroom': 17, 
              'concert_hall': 18, 'corridor': 19, 'deli': 20, 'dentaloffice': 21, 
              'dining_room': 22, 'elevator': 23, 'fastfood_restaurant': 24, 'florist': 25, 
              'gameroom': 26, 'garage': 27, 'greenhouse': 28, 'grocerystore': 29, 
              'gym': 30, 'hairsalon': 31, 'hospitalroom': 32, 'inside_bus': 33, 
              'inside_subway': 34, 'jewelleryshop': 35, 'kindergarden': 36, 'kitchen': 37, 
              'laboratorywet': 38, 'laundromat': 39, 'library': 40, 'livingroom': 41, 
              'lobby': 42, 'locker_room': 43, 'mall': 44, 'meeting_room': 45, 
              'movietheater': 46, 'museum': 47, 'nursery': 48, 'office': 49, 
              'operating_room': 50, 'pantry': 51, 'poolinside': 52, 'prisoncell': 53, 
              'restaurant': 54, 'restaurant_kitchen': 55, 'shoeshop': 56, 'stairscase': 57, 
              'studiomusic': 58, 'subway': 59, 'toystore': 60, 'trainstation': 61, 
              'tv_studio': 62, 'videostore': 63, 'waitingroom': 64, 'warehouse': 65, 
              'winecellar': 66}

cifar100_classes = {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4,
                    'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9,
                    'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14,
                    'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19,
                    'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24,
                    'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29,
                    'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34,
                    'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39,
                    'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44,
                    'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49,
                    'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54,
                    'otter': 55, 'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59,
                    'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64,
                    'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69,
                    'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74,
                    'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79,
                    'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84,
                    'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89,
                    'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94,
                    'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}

dtd_classes = {'banded': 0, 'blotchy': 1, 'braided': 2, 'bubbly': 3, 'bumpy': 4,
                'chequered': 5, 'cobwebbed': 6, 'cracked': 7, 'crosshatched': 8, 'crystalline': 9,
                'dotted': 10, 'fibrous': 11, 'flecked': 12, 'freckled': 13, 'frilly': 14,
                'gauzy': 15, 'grid': 16, 'grooved': 17, 'honeycombed': 18, 'interlaced': 19,
                'knitted': 20, 'lacelike': 21, 'lined': 22, 'marbled': 23, 'matted': 24,
                'meshed': 25, 'paisley': 26, 'perforated': 27, 'pitted': 28, 'pleated': 29,
                'polka-dotted': 30, 'porous': 31, 'potholed': 32, 'scaly': 33, 'smeared': 34, 
                'spiralled': 35, 'sprinkled': 36, 'stained': 37, 'stratified': 38, 'striped': 39,
                'studded': 40, 'swirly': 41, 'veined': 42, 'waffled': 43, 'woven': 44,
                'wrinkled': 45, 'zigzagged': 46}

caltech101_classes = {'Faces': 0, 'Faces_easy': 1, 'Leopards': 2, 'Motorbikes': 3, 'accordion': 4, 
                'airplanes': 5, 'anchor': 6, 'ant': 7, 'barrel': 8, 'bass': 9, 
                'beaver': 10, 'binocular': 11, 'bonsai': 12, 'brain': 13, 'brontosaurus': 14, 
                'buddha': 15, 'butterfly': 16, 'camera': 17, 'cannon': 18, 'car_side': 19,
                'ceiling_fan': 20, 'cellphone': 21, 'chair': 22, 'chandelier': 23, 'cougar_body': 24, 
                'cougar_face': 25, 'crab': 26, 'crayfish': 27, 'crocodile': 28, 'crocodile_head': 29, 
                'cup': 30, 'dalmatian': 31, 'dollar_bill': 32, 'dolphin': 33, 'dragonfly': 34, 
                'electric_guitar': 35, 'elephant': 36, 'emu': 37, 'euphonium': 38, 'ewer': 39,
                'ferry': 40, 'flamingo': 41, 'flamingo_head': 42, 'garfield': 43, 'gerenuk': 44,
                'gramophone': 45, 'grand_piano': 46, 'hawksbill': 47, 'headphone': 48, 'hedgehog': 49,
                'helicopter': 50, 'ibis': 51, 'inline_skate': 52, 'joshua_tree': 53, 'kangaroo': 54,
                'ketch': 55, 'lamp': 56, 'laptop': 57, 'llama': 58, 'lobster': 59,
                'lotus': 60, 'mandolin': 61,  'mayfly': 62, 'menorah': 63, 'metronome': 64,
                'minaret': 65, 'nautilus': 66, 'octopus': 67, 'okapi': 68, 'pagoda': 69,
                'panda': 70, 'pigeon': 71, 'pizza': 72, 'platypus': 73, 'pyramid': 74, 
                'revolver': 75, 'rhino': 76, 'rooster': 77, 'saxophone': 78, 'schooner': 79,
                'scissors': 80, 'scorpion': 81, 'sea_horse': 82, 'snoopy': 83, 'soccer_ball': 84,
                'stapler': 85, 'starfish': 86, 'stegosaurus': 87, 'stop_sign': 88, 'strawberry': 89,
                'sunflower': 90, 'tick': 91, 'trilobite': 92, 'umbrella': 93, 'watch': 94,
                'water_lilly': 95, 'wheelchair': 96, 'wild_cat': 97, 'windsor_chair': 98, 'wrench': 99,
                'yin_yang': 100
}

def main(args):

    save_path = os.path.join(args.syn_data_path, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == 'mit_indoor':
        model_id = "CompVis/stable-diffusion-v1-4"
        class_names = list(mit_classes.keys())
        prompt = lambda c: f"the inside of a {c}"
    elif args.dataset == 'cifar100' or args.dataset == 'caltech101':
        model_id = "CompVis/stable-diffusion-v1-4"
        class_names = list(cifar100_classes.keys())
        prompt = lambda c: f"a picture of a {c}"
    elif args.dataset == 'dtd':
        #model_id = "dream-textures/texture-diffusion"
        model_id = "CompVis/stable-diffusion-v1-4"
        class_names = list(dtd_classes.keys())
        prompt = lambda c: f"{c} texture"
    else:
        raise NotImplementedError(args.dataset)

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # Overrides and disables NSFW checker
    pipe.safety_checker = None
    pipe = pipe.to(f'cuda:{args.n_gpu}')

    for i in tqdm(range(len(class_names))):

        c = class_names[i]
        print(c)

        fpath = os.path.join(save_path, c)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        for j in range(args.num_img_per_class // args.batch_size):

            images = pipe(prompt(c), num_images_per_prompt=args.batch_size).images  

            for k, im in enumerate(images):
                im.save(f"{fpath}/{args.n_start + j * args.batch_size + k}.png")

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Generating Synthetic Samples')

    parser.add_argument('--dataset', default='mit_indoor', choices=['mit_indoor', 'cifar100', 'dtd', 'caltech101'], help='name of dataset')
    parser.add_argument('--num_img_per_class', type=int, default=80, help='number of images per class')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--n_start', type=int, default=0, help='index of sample to start at (non-zero if prev generated)')
    parser.add_argument('--n_gpu', type=int, default=0, help='index of gpu if multiple available')

    args = parser.parse_args()
    d = json.load(open('./paths.json'))
    vars(args).update(d)

    args.device = torch.device(f"cuda:{args.n_gpu}" if torch.cuda.is_available() else "cpu")
    print(args.device)

    main(args)
