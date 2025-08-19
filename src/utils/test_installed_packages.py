def run():
    import torch, pydicom, SimpleITK, nibabel, radiomics
    import pandas, numpy
    print("torch:", torch.__version__, " , cuda:", torch.version.cuda, " , gpu?", torch.cuda.is_available())
    print("pydicom OK, SimpleITK OK, nibabel OK, pyradiomics OK")

if __name__ == "__main__":
    run()

