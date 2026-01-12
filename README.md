# NST
Please put the **content images** into the [content](./data/content) folder, put the **style images** into the [style](./data/style) folder.

Before running, please use `.venv\Scripts\activate` to activate the virtual environment first.

Please use `python main.py --transfer gatysstyle --content YOUR_CONTENT_IMAGE.JPG --style YOUR_STYLE_IMAGE.JPG --show-image` to give your first try.

Now it has two available NST models, please use `--transfer` to specify which type you would choose, like`--transfer gatysstyle` for Gatys, `--transfer lapstyle` for Laplace.

You can also use `--size` to resize the size of your imgaes, use `--iterations` and `--lr` to adjust the hyperparameters.

Meanwhile, `--content-weight`, `style-weight` and `lap-weight` are also usable.

If you want to see the loss curve, please use `--plot-loss`. You could also use `--show-image` to see the original image and generated image immediately.

Finally, you are allowed to use `--compare` to print a picture to compare the results from different NST models. Use `--compare-mothods` to specify those models.

The outputs will be saved in [output](./output) folder, for gatys' in [gatysstyle](./output/gatysstyle), lap's in [lapstyle](./output/lapstyle) and comparison in [comparisons](./output/comparisons), including the generated image and the loss curves.
