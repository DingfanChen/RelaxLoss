import os
import sys
import argparse

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT_IMAGE = os.path.join(FILE_DIR, '../results/%s/%s/')
SAVE_ROOT_GENERAL = os.path.join(FILE_DIR, '../results/%s/')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name',
                        choices=['CIFAR10', 'CIFAR100', 'Texas', 'Purchase'])
    parser.add_argument('--model', type=str, default='resnet20', help='model architecture',
                        choices=['resnet20', 'vgg11_bn'])
    parser.add_argument('--method', type=str, default='relaxloss', help='method name',
                        choices=['vanilla', 'relaxloss', 'advreg', 'dpsgd', 'confidence_penalty', 'distillation',
                                 'dropout', 'early_stopping', 'label_smoothing'])
    parser.add_argument('--seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--mode', type=str, default='defense_attack', help='mode of the process to be run',
                        choices=['shadow', 'defense', 'attack', 'defense_attack'])
    args = parser.parse_args()
    return args


def run_defense(dataset_prefix, save_root, args, method):
    model_flag = (dataset_prefix == 'cifar')
    if method == 'distillation':
        ## train teacher model
        teacher_path = os.path.join(save_root, 'vanilla', f'seed{args.seed}')
        if not os.path.exists(os.path.join(teacher_path, 'model.pt')):
            command = f'python {dataset_prefix}/defense/vanilla.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset}'
            command += f' --model {args.model}' if model_flag else ''
            os.system(command)

        ## train student model
        command = f'python {dataset_prefix}/defense/{method}.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset} -teacher {teacher_path}'
        command += f' --model {args.model}' if model_flag else ''
        os.system(command)

    else:
        command = f'python {dataset_prefix}/defense/{method}.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset}'
        command += f' --model {args.model}' if model_flag else ''
        os.system(command)


def run_shadow(dataset_prefix, save_root, args, method):
    model_flag = (dataset_prefix == 'cifar')
    if method == 'distillation':
        ## train teacher model
        teacher_path = os.path.join(save_root, 'vanilla', f'seed{args.seed}', 'shadow')
        if not os.path.exists(os.path.join(teacher_path, 'model.pt')):
            command = f'python {dataset_prefix}/defense/vanilla.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset} --partition shadow'
            command += f' --model {args.model}' if model_flag else ''
            os.system(command)

        ## train student model
        command = f'python {dataset_prefix}/defense/{method}.py -name seed{args.seed} -s {args.seed} ' \
                  f'--dataset {args.dataset} -teacher {teacher_path} --partition shadow'
        command += f' --model {args.model}' if model_flag else ''
        os.system(command)

    else:
        command = f'python {dataset_prefix}/defense/{method}.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset} --partition shadow'
        command += f' --model {args.model}' if model_flag else ''
        os.system(command)


def run_attack(dataset_prefix, target, shadow):
    os.system(f'python {dataset_prefix}/run_attacks.py -target {target} -shadow {shadow}')


def main():
    args = parse_arguments()
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        dataset_prefix = 'cifar'
        save_root = SAVE_ROOT_IMAGE % (args.dataset, args.model)
    elif args.dataset in ['Texas', 'Purchase']:
        dataset_prefix = 'nonimage'
        save_root = SAVE_ROOT_GENERAL % args.dataset
    else:
        raise NotImplementedError
    if args.mode == 'defense':  ## only run defense
        run_defense(dataset_prefix, save_root, args, args.method)

    if args.mode == 'shadow':  ## only train shadow model
        run_shadow(dataset_prefix, save_root, args, args.method)

    if args.mode == 'attack':  ## only run attack
        ## train shadow model (for attack)
        base_shadow_path = os.path.join(save_root, 'vanilla', f'seed{args.seed}', 'shadow')
        if not os.path.exists(os.path.join(base_shadow_path, 'model.pt')):
            run_shadow(dataset_prefix, save_root, args, 'vanilla')

        ## run attack
        target_path = os.path.join(save_root, args.method, f'seed{args.seed}')
        if args.method == 'early_stopping':
            all_targets = [p for p in os.listdir(target_path) if os.path.isdir(os.path.join(target_path,p)) and 'ep' in p]
            for target_path in all_targets:
                run_attack(dataset_prefix, target_path, base_shadow_path)
        else:
            run_attack(dataset_prefix, target_path, base_shadow_path)

    if args.mode == 'defense_attack':  ## run both defense and attack
        ## run defense
        run_defense(dataset_prefix, save_root, args, args.method)

        ## train shadow model (for attack)
        base_shadow_path = os.path.join(save_root, 'vanilla', f'seed{args.seed}', 'shadow')
        if not os.path.exists(os.path.join(base_shadow_path, 'model.pt')):
            run_shadow(dataset_prefix, save_root, args, 'vanilla')

        ## run attack
        target_path = os.path.join(save_root, args.method, f'seed{args.seed}')
        if args.method == 'early_stopping':
            all_targets = [p for p in os.listdir(target_path) if os.path.isdir(p) and 'ep' in p]
            for target_path in all_targets:
                run_attack(dataset_prefix, target_path, base_shadow_path)
        else:
            run_attack(dataset_prefix, target_path, base_shadow_path)


if __name__ == '__main__':
    main()
