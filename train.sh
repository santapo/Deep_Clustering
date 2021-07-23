python cli.py --data.data_dir='data/no_label/images_mr' \
            --data.batch_size=1 \
            --data.num_workers=1 \
            --trainer.default_root_dir='exps/Vgg16/' \
            --trainer.max_epochs=200 \
            --trainer.val_check_interval=1.0 \
            --trainer.gpus=0 \
            --model.lr=0.001 \
            --model.optimizer_name='Adam' \


            
