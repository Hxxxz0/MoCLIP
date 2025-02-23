
from .t2m_dataset_true import HumanML3D,KIT

from os.path import join as pjoin
__all__ = [
    'HumanML3D', 'KIT',  'get_dataset',]

def get_dataset(opt, split='train', mode='train', accelerator=None):
    print('split:',split)
    if opt.dataset_name == 't2m' :
        dataset = HumanML3D(opt, split, mode, accelerator)
    elif opt.dataset_name == 'kit' :
        print('KIT')
        dataset = KIT(opt,split, mode, accelerator)
    else:
        raise KeyError('Dataset Does Not Exist')
    
    if accelerator:
        accelerator.print('Completing loading %s dataset' % opt.dataset_name)
    else:
        print('Completing loading %s dataset' % opt.dataset_name)
    
    return dataset

