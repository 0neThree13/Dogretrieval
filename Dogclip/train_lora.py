#使用lora微调
from lora import *
import clip
from Dog100K import *
from torch.utils.data import DataLoader


def main():

    # CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.backbone, device=device)
    model.eval()
    logit_scale = 100
    root_dir='Dog100K'
    seed = 42
    img_dataset = 'data'


    # Prepare dataset
    print("Preparing dataset.")


    train_set = Dog100K(root=root_dir, subset='train', img_dataset=img_dataset, seed=seed)
    test_set = Dog100K(root=root_dir, subset='test', img_dataset=img_dataset)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=256, collate_fn=collate_fn)


    run_lora(args,model,logit_scale,train_loader, test_loader)


if __name__ == '__main__':
    main()