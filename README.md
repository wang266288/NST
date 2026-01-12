# NST
Please put the **content images** into the [content](./data/content) folder, put the **style images** into the [style](./data/style) folder.

Before running, please use `.venv\Scripts\activate` to activate the virtual environment first.

Use `python main.py --content your_content_image.jpg --style your_style_image.jpg` to perform a style transfer operation.

You can also use `--size` to resize the size of your imgaes, use `--iterations` and `--lr` to adjust the hyperparameters.

The outputs will be saved in [output](./output) folder, including the generated image and the loss curves.
