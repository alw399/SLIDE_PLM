from glob import glob
import pandas as pd
from pathlib import Path


def get_sigLFs(slide_outs):
    '''
    Get the significant latent factors from the slide outputs
    @param slide_outs: path to the best performining slide outputs
    @return sigLFs: list of significant latent factors
    '''
    sig_LFs = glob(str(Path(slide_outs) / '*_list*'))
    sig_LFs = [f"Z{path.replace('.txt','').rsplit('Z', 1)[1]}" for path in sig_LFs]

    return sig_LFs

