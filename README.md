# FoMoMMGL Implementation

This is the implementation of **FoMoMMGL**, which adds the Graph Token module, combined with the **Structured MM Fusion Attention module, and ran experiments on the **WikiWeb2M** dataset. You can find the dataset at the following URL:

[https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md](https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md)

We would like to express our gratitude to the **MMGL** work for providing the codebase, which served as the baseline for our development. You can find MMGL here:

[https://github.com/minjiyoon/MMGL?tab=readme-ov-file](https://github.com/minjiyoon/MMGL?tab=readme-ov-file)

## How to Run

To run the code, you need to preprocess the data following the steps outlined in the MMGL work. After downloading and preparing the dataset, you can execute the following bash script:

```bash
sh script/train_generation.sh
```

## Future Work

Our work is under continuous development and improvement. We plan to expand it to support more language model backbones. Please stay tuned for updates on our GitHub repository:

[https://github.com/YennNing/FoMoMMGL](https://github.com/YennNing/FoMoMMGL)
