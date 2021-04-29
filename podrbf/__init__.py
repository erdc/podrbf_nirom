__all__ = [
    'main','greedy','pod','rbf', 'rom'
]

__project__ = 'Greedy-POD-RBF-Nirom'
__title__ = "podrbf"
__author__ = "Sourav Dutta, Matthew Farthing"
__copyright__ = "Copyright 2019-2020, PODRBF contributors"
__license__ = "MIT"
__version__ = "1.0"
__mail__ = 'sourav.dutta@erdc.dren.mil, matthew.w.farthing@erdc.dren.mil'
__maintainer__ = __author__
__status__ = "Stable"

from .main import PODRBFBase
from .pod import pod
from .rbf import rbf
from .rom import rom
from .greedy import greedy
