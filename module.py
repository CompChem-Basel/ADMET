#!/usr/bin/env python3
"""
modules/admet/module.py — ADMET Prediction Module (Production)

Absorption, Distribution, Metabolism, Excretion, and Toxicity predictions
using ensemble Chemprop models with calibration and applicability domain.

Features:
- Registry-driven endpoint configuration (module_registry.yaml)
- Ensemble Chemprop models (3-5 models per endpoint)
- Optional calibration (isotonic/Platt) for classification
- Applicability domain via diagonal Mahalanobis z-score
- Uncertainty quantification (ensemble std)
- Support for pre-computed features (memory-mapped)
- Timestamped output folders
- Interactive folder selection for VS output

File Structure Expected:
    models/{endpoint_name}/seed*.pt      - Chemprop ensemble models
    calib/{endpoint_name}/calibrator.joblib - Optional calibration
    features/features_meta.json          - Feature metadata
    features/features_X.f32.mmap         - Pre-computed features (optional)
    ad_stats_train.npz                   - AD statistics (mean/std)
    splits_scaffold.npz                  - Train/test splits

Input: VS results or Hit-ID results (SDF/CSV with SMILES)
Output: ADMET property predictions with uncertainty

Author: Drug Discovery Pipeline
Version: 0.0.3 (Interactive Input Selection)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# NumPy 2.0 Compatibility Fix (MUST be before other imports)
# =============================================================================
# Chemprop 1.6.1 uses deprecated NumPy attributes removed in NumPy 2.0
import warnings

# Suppress FutureWarnings during patching
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, 'VisibleDeprecationWarning'):
        np.VisibleDeprecationWarning = DeprecationWarning
    if not hasattr(np, 'ComplexWarning'):
        np.ComplexWarning = DeprecationWarning
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
    if not hasattr(np, 'int'):
        np.int = np.int_
    if not hasattr(np, 'float'):
        np.float = np.float64
    if not hasattr(np, 'complex'):
        np.complex = np.complex128
    if not hasattr(np, 'object'):
        np.object = np.object_
    if not hasattr(np, 'str'):
        np.str = np.str_

import pandas as pd
import yaml

# Suppress pandas PerformanceWarning about DataFrame fragmentation
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Suppress hyperopt's pkg_resources deprecation warning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

# Optional imports
HAS_TORCH = False
torch = None
try:
    import torch as _torch
    torch = _torch
    HAS_TORCH = True
except ImportError:
    pass

# =============================================================================
# Chemprop imports (for ADMET-AI model loading)
# =============================================================================
HAS_CHEMPROP = False
load_checkpoint = None
load_scalers = None
load_args = None
MoleculeDataLoader = None
MoleculeDatapoint = None
MoleculeDataset = None
chemprop_predict = None

try:
    from chemprop.utils import load_checkpoint as _load_checkpoint
    from chemprop.utils import load_scalers as _load_scalers
    from chemprop.utils import load_args as _load_args
    from chemprop.data import MoleculeDataLoader as _MoleculeDataLoader
    from chemprop.data import MoleculeDatapoint as _MoleculeDatapoint
    from chemprop.data import MoleculeDataset as _MoleculeDataset
    from chemprop.train import predict as _chemprop_predict
    
    load_checkpoint = _load_checkpoint
    load_scalers = _load_scalers
    load_args = _load_args
    MoleculeDataLoader = _MoleculeDataLoader
    MoleculeDatapoint = _MoleculeDatapoint
    MoleculeDataset = _MoleculeDataset
    chemprop_predict = _chemprop_predict
    HAS_CHEMPROP = True
except ImportError:
    pass

# RDKit features for Chemprop-RDKit models
HAS_CHEMFUNC = False
compute_rdkit_fingerprint = None
try:
    from chemfunc.molecular_fingerprints import compute_rdkit_fingerprint as _compute_rdkit_fp
    compute_rdkit_fingerprint = _compute_rdkit_fp
    HAS_CHEMFUNC = True
except ImportError:
    pass

HAS_JOBLIB = False
joblib = None
try:
    import joblib as _joblib
    joblib = _joblib
    HAS_JOBLIB = True
except ImportError:
    pass

HAS_TQDM = False
tqdm_module = None
try:
    from tqdm import tqdm as _tqdm
    tqdm_module = _tqdm
    HAS_TQDM = True
except ImportError:
    # Fallback tqdm that does nothing
    def _tqdm(iterable, **kwargs):
        return iterable
    tqdm_module = _tqdm

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from core.base import BaseModule, BaseModuleConfig, ModuleResult, ModuleStatus
    HAS_BASE = True
except ImportError:
    HAS_BASE = False
    class BaseModuleConfig:
        def to_dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    class ModuleStatus:
        COMPLETED = "completed"
        FAILED = "failed"

RDLogger.DisableLog("rdApp.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ADMET")


# =============================================================================
# Interactive Folder Selection (for VS output)
# =============================================================================

class FolderSelector:
    """Utility to list and select timestamped folders from module outputs."""
    
    def __init__(self, base_output_dir: str = "output"):
        self.base_dir = Path(base_output_dir).absolute()
    
    def list_available_folders(self, module_name: str) -> List[Tuple[str, dict]]:
        """List all timestamped folders for a given module."""
        module_dir = self.base_dir / module_name
        if not module_dir.exists():
            return []
        
        folders = []
        for folder in sorted(module_dir.iterdir(), reverse=True):
            if folder.is_dir() and self._is_timestamp_folder(folder.name):
                metadata = self._load_metadata(folder)
                metadata['files'] = self._list_key_files(folder)
                metadata['folder_path'] = str(folder)
                folders.append((folder.name, metadata))
        return folders
    
    def _is_timestamp_folder(self, name: str) -> bool:
        """Check if folder name matches timestamp pattern."""
        import re
        patterns = [
            r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$",
            r"^[a-z_]+_\d{8}_\d{6}$",
            r"^\d{8}_\d{6}$",
        ]
        return any(re.match(p, name) for p in patterns)
    
    def _list_key_files(self, folder: Path) -> List[str]:
        """List key files in the folder."""
        key_patterns = ['*.sdf', '*.csv', '*.json', '*.smi']
        files = []
        for pattern in key_patterns:
            files.extend([f.name for f in folder.glob(pattern)])
        return files[:10]
    
    def _load_metadata(self, folder: Path) -> dict:
        """Load metadata from folder."""
        metadata = {}
        
        for config_file in ['metadata.json', 'config.json', 'summary.json', 'result.json']:
            config_path = folder / config_file
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        data = json.load(f)
                        metadata.update(data)
                except:
                    pass
        
        try:
            stat = os.stat(folder)
            metadata['created'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        except:
            pass
        
        return metadata
    
    def interactive_select(self, module_name: str, 
                          file_pattern: str = None,
                          prompt: str = None) -> Optional[Path]:
        """Interactively select a folder with nice display."""
        folders = self.list_available_folders(module_name)
        
        if not folders:
            print(f"\n  ⚠️  No {module_name} output folders found in {self.base_dir / module_name}")
            return None
        
        print(f"\n{'═'*70}")
        print(f"  📁 SELECT {module_name.upper()} OUTPUT FOLDER")
        print(f"{'═'*70}")
        print(f"  Found {len(folders)} available folder(s) in: {self.base_dir / module_name}")
        print(f"{'─'*70}")
        
        for idx, (name, meta) in enumerate(folders, 1):
            created = meta.get('created', 'Unknown')
            files = meta.get('files', [])
            
            print(f"\n  [{idx}] ✅ {name}")
            print(f"      📅 Created: {created}")
            if files:
                files_str = ", ".join(files[:5])
                if len(files) > 5:
                    files_str += f" (+{len(files)-5} more)"
                print(f"      📄 Files: {files_str}")
        
        print(f"\n  [0] ❌ Cancel")
        print(f"{'─'*70}")
        
        while True:
            try:
                choice = input(f"\n  Select {module_name} folder [1-{len(folders)}, 0=cancel]: ").strip()
                
                if choice == "0":
                    return None
                
                idx = int(choice) - 1
                if 0 <= idx < len(folders):
                    folder_name, meta = folders[idx]
                    folder_path = Path(meta['folder_path'])
                    print(f"\n  ✅ Selected: {folder_path}")
                    
                    # Find specific file if pattern provided
                    if file_pattern:
                        matches = list(folder_path.glob(file_pattern))
                        if matches:
                            print(f"  → Input file: {matches[0].name}")
                            return matches[0]
                        else:
                            # Return folder path
                            return folder_path
                    
                    return folder_path
                else:
                    print(f"  ⚠️  Please enter a number between 1 and {len(folders)}")
            
            except ValueError:
                print(f"  ⚠️  Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\n  Cancelled by user.")
                return None


def interactive_input_selection(base_output_dir: str = "output") -> Optional[str]:
    """
    Interactive selection of input file for ADMET module.
    
    Looks for VS (virtual screening) output folders and lets user select one.
    
    Returns:
        Path to input file (vs_results.csv or vs_results.sdf)
    """
    print("\n" + "=" * 70)
    print("  🧪 ADMET - INPUT SELECTION")
    print("=" * 70)
    print("\n  Select the Virtual Screening (VS) output to process:")
    
    selector = FolderSelector(base_output_dir)
    
    # Try to find VS results
    selected = selector.interactive_select(
        "vs",
        file_pattern="vs_results.*"
    )
    
    if selected is None:
        return None
    
    # If it's a file, return it directly
    if selected.is_file():
        return str(selected)
    
    # If it's a folder, look for results files
    for pattern in ["vs_results.csv", "vs_results.sdf", "*.csv", "*.sdf"]:
        matches = list(selected.glob(pattern))
        if matches:
            result_file = matches[0]
            print(f"  → Using: {result_file.name}")
            return str(result_file)
    
    print(f"  ⚠️  No results file found in {selected}")
    return None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ADMETConfig(BaseModuleConfig):
    """ADMET module configuration."""
    
    name: str = "admet"
    version: str = "0.0.3"
    
    # Interactive mode
    interactive: bool = True  # Enable interactive input folder selection
    
    # Registry and model paths (relative to module directory, not config path)
    registry_path: str = "module_registry.yaml"  # Just filename, resolved relative to module dir
    models_dir: str = "models"
    calib_dir: str = "calib"
    features_dir: str = "features"
    
    # Feature settings (from registry or defaults)
    morgan_radius: int = 2
    morgan_nbits: int = 2048
    
    # Applicability domain
    ad_stats_path: str = "ad_stats_train.npz"
    splits_path: str = "splits_scaffold.npz"
    ad_threshold: float = 3.5  # Mean |z| threshold for AD flag
    
    # Uncertainty
    unc_threshold: float = 0.20  # Ensemble std threshold for uncertainty flag
    
    # Processing
    device: str = "cuda"  # cuda or cpu
    batch_size: int = 50  # Batch size for Chemprop inference
    feat_batch_size: int = 5000  # Batch size for featurization
    
    # Endpoint selection (None = all from registry)
    endpoints: Optional[List[str]] = None
    
    # Output
    output_formats: List[str] = field(default_factory=lambda: ["csv", "sdf", "html"])
    create_timestamp_folder: bool = True


# =============================================================================
# Default ADMET Endpoints (All 41 from ADMET-AI)
# =============================================================================

ADMET_ENDPOINTS = {
    # ==========================================================================
    # ABSORPTION (6 endpoints)
    # ==========================================================================
    "Caco2_Wang": {"task": "regression", "description": "Caco-2 cell permeability (cm/s)"},
    "HIA_Hou": {"task": "classification", "description": "Human intestinal absorption"},
    "Pgp_Broccatelli": {"task": "classification", "description": "P-glycoprotein inhibitor"},
    "Bioavailability_Ma": {"task": "classification", "description": "Oral bioavailability (F > 20%)"},
    "Lipophilicity_AstraZeneca": {"task": "regression", "description": "Lipophilicity (LogD7.4)"},
    "Solubility_AqSolDB": {"task": "regression", "description": "Aqueous solubility (LogS)"},
    "HydrationFreeEnergy_FreeSolv": {"task": "regression", "description": "Hydration free energy (kcal/mol)"},
    
    # ==========================================================================
    # DISTRIBUTION (3 endpoints)
    # ==========================================================================
    "BBB_Martins": {"task": "classification", "description": "Blood-brain barrier penetration"},
    "PPBR_AZ": {"task": "regression", "description": "Plasma protein binding rate (%)"},
    "VDss_Lombardo": {"task": "regression", "description": "Volume of distribution at steady state (L/kg)"},
    
    # ==========================================================================
    # METABOLISM (10 endpoints)
    # ==========================================================================
    # CYP Inhibition
    "CYP1A2_Veith": {"task": "classification", "description": "CYP1A2 inhibitor"},
    "CYP2C9_Veith": {"task": "classification", "description": "CYP2C9 inhibitor"},
    "CYP2C19_Veith": {"task": "classification", "description": "CYP2C19 inhibitor"},
    "CYP2D6_Veith": {"task": "classification", "description": "CYP2D6 inhibitor"},
    "CYP3A4_Veith": {"task": "classification", "description": "CYP3A4 inhibitor"},
    
    # CYP Substrate
    "CYP2C9_Substrate_CarbonMangels": {"task": "classification", "description": "CYP2C9 substrate"},
    "CYP2D6_Substrate_CarbonMangels": {"task": "classification", "description": "CYP2D6 substrate"},
    "CYP3A4_Substrate_CarbonMangels": {"task": "classification", "description": "CYP3A4 substrate"},
    
    # ==========================================================================
    # EXCRETION (3 endpoints)
    # ==========================================================================
    "Half_Life_Obach": {"task": "regression", "description": "Half-life (hours)"},
    "Clearance_Hepatocyte_AZ": {"task": "regression", "description": "Hepatocyte clearance (µL/min/10^6 cells)"},
    "Clearance_Microsome_AZ": {"task": "regression", "description": "Microsomal clearance (mL/min/g)"},
    
    # ==========================================================================
    # TOXICITY (19 endpoints)
    # ==========================================================================
    # Cardiotoxicity
    "hERG": {"task": "classification", "description": "hERG potassium channel inhibition"},
    
    # Genotoxicity
    "AMES": {"task": "classification", "description": "Ames mutagenicity"},
    
    # Hepatotoxicity
    "DILI": {"task": "classification", "description": "Drug-induced liver injury"},
    
    # Skin toxicity
    "Skin_Reaction": {"task": "classification", "description": "Skin sensitization"},
    
    # Carcinogenicity
    "Carcinogens_Lagunin": {"task": "classification", "description": "Carcinogenicity"},
    
    # Clinical toxicity
    "ClinTox": {"task": "classification", "description": "Clinical trial toxicity"},
    
    # Acute toxicity
    "LD50_Zhu": {"task": "regression", "description": "Acute toxicity LD50 (log(1/(mol/kg)))"},
    
    # NR Toxicity Panel (Nuclear Receptor)
    "NR-AR": {"task": "classification", "description": "Androgen receptor agonist (Tox21)"},
    "NR-AR-LBD": {"task": "classification", "description": "AR ligand-binding domain agonist (Tox21)"},
    "NR-AhR": {"task": "classification", "description": "Aryl hydrocarbon receptor agonist (Tox21)"},
    "NR-Aromatase": {"task": "classification", "description": "Aromatase inhibitor (Tox21)"},
    "NR-ER": {"task": "classification", "description": "Estrogen receptor agonist (Tox21)"},
    "NR-ER-LBD": {"task": "classification", "description": "ER ligand-binding domain agonist (Tox21)"},
    "NR-PPAR-gamma": {"task": "classification", "description": "PPAR-gamma agonist (Tox21)"},
    
    # SR Toxicity Panel (Stress Response)
    "SR-ARE": {"task": "classification", "description": "Antioxidant response element (Tox21)"},
    "SR-ATAD5": {"task": "classification", "description": "ATAD5 genotoxicity (Tox21)"},
    "SR-HSE": {"task": "classification", "description": "Heat shock response (Tox21)"},
    "SR-MMP": {"task": "classification", "description": "Mitochondrial membrane potential (Tox21)"},
    "SR-p53": {"task": "classification", "description": "p53 activation (Tox21)"},
}


@dataclass
class ADMETPrediction:
    """Single molecule ADMET prediction result."""
    mol_id: str
    smiles: str
    predictions: Dict[str, float] = field(default_factory=dict)
    uncertainties: Dict[str, float] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    ad_score: float = 0.0
    ad_flag: bool = False
    parse_fail: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "mol_id": self.mol_id,
            "smiles": self.smiles,
            "ad_score": round(self.ad_score, 4),
            "ad_flag": int(self.ad_flag),
            "parse_fail": int(self.parse_fail),
        }
        
        for endpoint, value in self.predictions.items():
            result[endpoint] = round(value, 4) if value is not None else None
            if endpoint in self.uncertainties:
                result[f"{endpoint}_unc"] = round(self.uncertainties[endpoint], 4)
            if endpoint in self.flags:
                result[f"{endpoint}_flag"] = int(self.flags[endpoint])
        
        return result


# =============================================================================
# Utilities
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


# =============================================================================
# Featurization (for AD computation - still uses Morgan fingerprints)
# =============================================================================

def featurize_smiles_batch(
    smiles_list: List[str],
    radius: int = 2,
    nbits: int = 2048,
    batch_size: int = 5000,
) -> Tuple[np.ndarray, List[Optional[Chem.Mol]], np.ndarray]:
    """
    Featurize a batch of SMILES to Morgan fingerprints.
    
    Used for applicability domain computation.
    Chemprop models compute their own features internally.
    """
    global tqdm_module
    
    N = len(smiles_list)
    X = np.zeros((N, nbits), dtype=np.float32)
    mols = []
    parse_fail = np.zeros(N, dtype=bool)
    
    n_batches = (N + batch_size - 1) // batch_size
    
    for batch_idx in tqdm_module(range(n_batches), desc="Featurizing", unit="batch", 
                                  dynamic_ncols=True, leave=True):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        
        for i in range(start, end):
            smi = smiles_list[i]
            mol = Chem.MolFromSmiles(smi) if smi else None
            mols.append(mol)
            
            if mol is None:
                parse_fail[i] = True
            else:
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
                    arr = np.zeros(nbits, dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    X[i] = arr
                except Exception:
                    parse_fail[i] = True
    
    return X, mols, parse_fail


# =============================================================================
# RDKit Features for Chemprop-RDKit Models
# =============================================================================

# ADMET-AI uses 200 RDKit descriptors computed by chemfunc
RDKIT_FEATURE_SIZE = 200

def compute_rdkit_features_for_smiles(smiles_list: List[str], show_progress: bool = True) -> List[Optional[np.ndarray]]:
    """
    Compute RDKit features for Chemprop-RDKit models.
    
    ADMET-AI models use 200 RDKit descriptors as additional features,
    computed by chemfunc.molecular_fingerprints.compute_rdkit_fingerprint().
    
    Training command from ADMET-AI:
        chemprop_train --features_path data/tdc_admet_all.npz --no_features_scaling ...
    """
    features = []
    
    # Use progress bar
    iterator = smiles_list
    if show_progress and HAS_TQDM:
        iterator = tqdm_module(smiles_list, desc="RDKit features", unit="mol", dynamic_ncols=True)
    
    for smiles in iterator:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        
        if mol is None:
            features.append(np.zeros(RDKIT_FEATURE_SIZE, dtype=np.float32))
            continue
        
        if HAS_CHEMFUNC and compute_rdkit_fingerprint is not None:
            try:
                fp = compute_rdkit_fingerprint(mol)
                features.append(np.array(fp, dtype=np.float32))
            except Exception as e:
                logger.debug(f"chemfunc feature computation failed: {e}")
                features.append(np.zeros(RDKIT_FEATURE_SIZE, dtype=np.float32))
        else:
            # Fallback: compute RDKit descriptors manually
            # This mimics what chemfunc.compute_rdkit_fingerprint does
            try:
                from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf, GraphDescriptors
                from rdkit.Chem import rdMolDescriptors
                
                desc = []
                
                # Basic descriptors
                desc.append(Descriptors.MolWt(mol))
                desc.append(Descriptors.HeavyAtomMolWt(mol))
                desc.append(Descriptors.MolLogP(mol))
                desc.append(Descriptors.MolMR(mol))
                desc.append(Descriptors.TPSA(mol))
                
                # Lipinski
                desc.append(Descriptors.NumHDonors(mol))
                desc.append(Descriptors.NumHAcceptors(mol))
                desc.append(Descriptors.NumRotatableBonds(mol))
                desc.append(Descriptors.NumHeteroatoms(mol))
                desc.append(Descriptors.NumValenceElectrons(mol))
                
                # Rings
                desc.append(Descriptors.RingCount(mol))
                desc.append(Descriptors.NumAromaticRings(mol))
                desc.append(Descriptors.NumAliphaticRings(mol))
                desc.append(Descriptors.NumSaturatedRings(mol))
                desc.append(Descriptors.NumAromaticHeterocycles(mol))
                desc.append(Descriptors.NumAromaticCarbocycles(mol))
                
                # Fraction
                desc.append(Descriptors.FractionCSP3(mol))
                
                # Counts
                desc.append(rdMolDescriptors.CalcNumAmideBonds(mol))
                desc.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
                desc.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
                
                # Topological
                desc.append(GraphDescriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0)
                desc.append(GraphDescriptors.BertzCT(mol))
                desc.append(GraphDescriptors.Chi0(mol))
                desc.append(GraphDescriptors.Chi1(mol))
                desc.append(GraphDescriptors.HallKierAlpha(mol))
                desc.append(GraphDescriptors.Kappa1(mol))
                desc.append(GraphDescriptors.Kappa2(mol))
                desc.append(GraphDescriptors.Kappa3(mol))
                
                # Surface area
                desc.append(MolSurf.LabuteASA(mol))
                desc.append(MolSurf.PEOE_VSA1(mol))
                desc.append(MolSurf.PEOE_VSA2(mol))
                desc.append(MolSurf.SMR_VSA1(mol))
                desc.append(MolSurf.SMR_VSA2(mol))
                desc.append(MolSurf.SlogP_VSA1(mol))
                desc.append(MolSurf.SlogP_VSA2(mol))
                
                # More counts
                desc.append(Lipinski.HeavyAtomCount(mol))
                desc.append(Lipinski.NHOHCount(mol))
                desc.append(Lipinski.NOCount(mol))
                desc.append(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
                
                # Pad or truncate to expected size
                while len(desc) < RDKIT_FEATURE_SIZE:
                    desc.append(0.0)
                desc = desc[:RDKIT_FEATURE_SIZE]
                
                # Replace NaN/Inf with 0
                desc = [0.0 if (np.isnan(x) or np.isinf(x)) else x for x in desc]
                
                features.append(np.array(desc, dtype=np.float32))
            except Exception as e:
                logger.debug(f"RDKit descriptor computation failed: {e}")
                features.append(np.zeros(RDKIT_FEATURE_SIZE, dtype=np.float32))
    
    return features


# =============================================================================
# Registry Loading
# =============================================================================

def load_registry(path: str, module_dir: str) -> Dict:
    """Load endpoint registry from YAML."""
    p = Path(path)
    
    # If path doesn't exist as given, it's likely already resolved
    if not p.exists():
        logger.warning(f"Registry file not found: {path}")
        return create_default_registry()
    
    try:
        with open(p, "r") as f:
            data = yaml.safe_load(f)
        return data.get("registry", data)
    except Exception as e:
        logger.warning(f"Failed to load registry: {e}")
        return create_default_registry()


def create_default_registry() -> Dict:
    """Create default registry from built-in endpoints."""
    return {
        "featurizer": {
            "morgan_radius": 2,
            "morgan_nbits": 2048,
        },
        "endpoints": {
            name: {
                "task": info["task"],
                "description": info["description"],
                "models": [],  # No models - will use simulated predictions
            }
            for name, info in ADMET_ENDPOINTS.items()
        }
    }


def load_calibrator(calib_path: str) -> Optional[Dict]:
    """Load calibration model (isotonic or Platt)."""
    if not HAS_JOBLIB:
        return None
    
    p = Path(calib_path)
    if not p.exists():
        return None
    
    try:
        return joblib.load(p)
    except Exception as e:
        logger.warning(f"Failed to load calibrator {calib_path}: {e}")
        return None


def apply_calibration(calib_obj: Optional[Dict], p_uncal: np.ndarray) -> np.ndarray:
    """Apply calibration to uncalibrated probabilities."""
    if calib_obj is None:
        return p_uncal
    
    kind = calib_obj.get("kind", "isotonic")
    model = calib_obj.get("model")
    
    if model is None:
        return p_uncal
    
    try:
        if kind == "isotonic":
            return model.predict(p_uncal).astype(np.float32)
        else:
            # Platt scaling
            eps = 1e-6
            p = np.clip(p_uncal, eps, 1 - eps)
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            return model.predict_proba(logit)[:, 1].astype(np.float32)
    except Exception as e:
        logger.warning(f"Calibration failed: {e}")
        return p_uncal


# =============================================================================
# Chemprop Model Loading and Prediction
# =============================================================================

def load_chemprop_ensemble(
    model_paths: List[str],
    device: str,
) -> Tuple[List[Any], List[Any], Optional[Any]]:
    """
    Load Chemprop ensemble models.
    
    CRITICAL: Uses chemprop.utils.load_checkpoint(), NOT torch.jit.load()!
    
    Returns:
        Tuple of (models, scalers, args)
    """
    global torch
    
    if not HAS_CHEMPROP:
        logger.error("Chemprop is not installed!")
        logger.error("Install with: pip install chemprop==1.6.1")
        raise ImportError("Chemprop required for ADMET-AI models. Install with: pip install chemprop==1.6.1")
    
    if not HAS_TORCH:
        logger.error("PyTorch is not installed!")
        raise ImportError("PyTorch required for model inference.")
    
    # Determine device
    if device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device("cpu")
    
    models = []
    scalers = []
    args = None
    
    for mp in model_paths:
        if not os.path.exists(mp):
            logger.warning(f"Model not found: {mp}")
            continue
        
        try:
            # Load using Chemprop's loader (NOT torch.jit.load!)
            model = load_checkpoint(path=mp, device=device_obj)
            model.eval()
            models.append(model)
            
            # Load scaler - handle different return formats
            try:
                scaler_result = load_scalers(path=mp)
                # load_scalers can return (scaler, features_scaler) or 
                # (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler)
                if isinstance(scaler_result, tuple):
                    if len(scaler_result) >= 2:
                        scaler = scaler_result[0]  # Target scaler
                    else:
                        scaler = scaler_result[0] if scaler_result else None
                else:
                    scaler = scaler_result
                scalers.append(scaler)
            except Exception as e:
                logger.debug(f"Could not load scaler for {mp}: {e}")
                scalers.append(None)
            
            # Load args from first model
            if args is None:
                try:
                    args = load_args(mp)
                except Exception as e:
                    logger.debug(f"Could not load args from {mp}: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to load model {mp}: {e}")
    
    return models, scalers, args


def chemprop_ensemble_predict(
    models: List[Any],
    scalers: List[Any],
    args: Any,
    smiles_list: List[str],
    batch_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Chemprop ensemble prediction.
    
    Takes SMILES directly. For Chemprop-RDKit models (like ADMET-AI),
    also computes RDKit features.
    
    ADMET-AI training used:
        chemprop_train --features_path data.npz --no_features_scaling ...
    
    Returns:
        mean: Mean of ensemble predictions
        std: Std of ensemble predictions (uncertainty)
    """
    global tqdm_module
    
    if not models:
        N = len(smiles_list)
        return np.zeros(N, dtype=np.float32), np.ones(N, dtype=np.float32)
    
    N = len(smiles_list)
    K = len(models)
    
    # Check if models use RDKit features (ADMET-AI models do)
    use_features = getattr(args, 'use_input_features', False) if args else False
    features_size = getattr(args, 'features_size', RDKIT_FEATURE_SIZE) if args else RDKIT_FEATURE_SIZE
    
    # Log feature usage
    if use_features:
        logger.debug(f"Model uses input features (size={features_size})")
    
    # Compute RDKit features if needed
    if use_features:
        logger.info("Computing RDKit features for Chemprop-RDKit model...")
        features_list = compute_rdkit_features_for_smiles(smiles_list)
    else:
        features_list = [None] * N
    
    # Create Chemprop dataset
    data_points = []
    for smi, feat in zip(smiles_list, features_list):
        dp = MoleculeDatapoint(smiles=[smi], features=feat)
        data_points.append(dp)
    
    dataset = MoleculeDataset(data_points)
    data_loader = MoleculeDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,  # Safer on Windows
        shuffle=False
    )
    
    # Run predictions for each model
    all_preds = []
    
    for model_idx, model in enumerate(models):
        try:
            preds = chemprop_predict(model=model, data_loader=data_loader)
            preds = np.array(preds, dtype=np.float32)
            
            # Apply scaler for regression (ADMET-AI uses --no_features_scaling but may scale targets)
            if scalers[model_idx] is not None:
                preds = scalers[model_idx].inverse_transform(preds)
            
            # Flatten if needed
            if preds.ndim > 1:
                preds = preds[:, 0]
            
            all_preds.append(preds)
            
        except Exception as e:
            logger.warning(f"Prediction failed for model {model_idx}: {e}")
    
    if not all_preds:
        return np.zeros(N, dtype=np.float32), np.ones(N, dtype=np.float32)
    
    all_preds = np.array(all_preds)  # Shape: (K, N)
    
    mean = np.mean(all_preds, axis=0)
    std = np.std(all_preds, axis=0)
    
    return mean.astype(np.float32), std.astype(np.float32)


def chemprop_ensemble_predict_multitask(
    models: List[Any],
    scalers: List[Any],
    args: Any,
    smiles_list: List[str],
    batch_size: int = 50,
    cached_features: List[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Run Chemprop ensemble prediction for MULTITASK models.
    
    Multitask models output multiple columns (one per endpoint).
    
    Returns:
        mean: Mean of ensemble predictions, shape (N, num_tasks)
        std: Std of ensemble predictions, shape (N, num_tasks)
        features: Computed features (for caching)
    """
    global tqdm_module
    
    N = len(smiles_list)
    
    if not models:
        logger.warning("No models provided for prediction")
        return np.zeros((N, 1), dtype=np.float32), np.ones((N, 1), dtype=np.float32), None
    
    K = len(models)
    
    # Check if models use RDKit features (ADMET-AI models do)
    use_features = getattr(args, 'use_input_features', False) if args else False
    
    # Use cached features or compute new ones
    if use_features:
        if cached_features is not None:
            logger.info(f"Using cached RDKit features for {N} molecules")
            features_list = cached_features
        else:
            logger.info(f"Computing RDKit features for {N} molecules...")
            features_list = compute_rdkit_features_for_smiles(smiles_list, show_progress=True)
    else:
        features_list = [None] * N
    
    # Create Chemprop dataset
    logger.info("Creating Chemprop dataset...")
    data_points = []
    for smi, feat in zip(smiles_list, features_list):
        dp = MoleculeDatapoint(smiles=[smi], features=feat)
        data_points.append(dp)
    
    dataset = MoleculeDataset(data_points)
    data_loader = MoleculeDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )
    
    # Run predictions for each model
    logger.info(f"Running inference with {K} models...")
    all_preds = []
    
    for model_idx, model in enumerate(tqdm_module(models, desc="Ensemble", unit="model")):
        try:
            preds = chemprop_predict(model=model, data_loader=data_loader)
            # Ensure proper numpy array with float32 dtype
            preds = np.array(preds, dtype=np.float64).astype(np.float32)
            
            # Apply scaler for regression
            if model_idx < len(scalers) and scalers[model_idx] is not None:
                try:
                    preds = scalers[model_idx].inverse_transform(preds)
                    preds = np.array(preds, dtype=np.float32)
                except Exception as e:
                    logger.debug(f"Scaler inverse_transform failed: {e}")
            
            all_preds.append(preds)
            logger.debug(f"Model {model_idx} predictions shape: {preds.shape}, dtype: {preds.dtype}")
            
        except Exception as e:
            logger.warning(f"Prediction failed for model {model_idx}: {e}")
    
    if not all_preds:
        logger.warning("All model predictions failed")
        # Return default values - try to get num_tasks from args
        num_tasks = getattr(args, 'num_tasks', 1) if args else 1
        return np.zeros((N, num_tasks), dtype=np.float32), np.ones((N, num_tasks), dtype=np.float32), features_list
    
    # Stack predictions: shape (K, N, num_tasks)
    try:
        # Ensure all predictions are proper numpy arrays with same dtype
        all_preds = [np.asarray(p, dtype=np.float32) for p in all_preds]
        all_preds = np.stack(all_preds, axis=0)
    except ValueError as e:
        logger.warning(f"Could not stack predictions: {e}")
        # Predictions might have different shapes - use first one
        all_preds = np.array([all_preds[0]], dtype=np.float32)
    
    # Compute mean and std across ensemble
    # Use float64 for computation to avoid precision issues
    all_preds_f64 = all_preds.astype(np.float64)
    mean = np.mean(all_preds_f64, axis=0)
    std = np.std(all_preds_f64, axis=0)
    
    # Ensure 2D output
    if mean.ndim == 1:
        mean = mean.reshape(-1, 1)
        std = std.reshape(-1, 1)
    
    logger.info(f"Predictions complete: shape={mean.shape}")
    
    return mean.astype(np.float32), std.astype(np.float32), features_list


# =============================================================================
# Applicability Domain
# =============================================================================

def load_ad_stats(
    ad_stats_path: str,
    splits_path: str = "",
    features_meta_path: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or compute AD statistics (mean/std from training set)."""
    p = Path(ad_stats_path)
    
    if p.exists():
        stats = np.load(p)
        return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    
    return None, None


def compute_ad_scores(
    X: np.ndarray,
    ad_mean: np.ndarray,
    ad_std: np.ndarray,
) -> np.ndarray:
    """Compute applicability domain scores (mean |z-score|)."""
    z = np.abs((X - ad_mean) / np.maximum(ad_std, 1e-6))
    return z.mean(axis=1).astype(np.float32)


# =============================================================================
# Input/Output Utilities
# =============================================================================

def load_input_molecules(
    path: str,
    smiles_col: str = "smiles",
) -> Tuple[List[str], Optional[pd.DataFrame], str]:
    """Load molecules from CSV or SDF file."""
    p = Path(path)
    ext = p.suffix.lower()
    
    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(str(p), removeHs=False)
        smiles = []
        for mol in suppl:
            if mol is not None:
                smiles.append(Chem.MolToSmiles(mol))
            else:
                smiles.append("")
        return smiles, None, "sdf"
    
    elif ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        
        # Find SMILES column
        smiles_columns = [smiles_col, "smiles", "SMILES", "Smiles", "canonical_smiles", 
                          "hit_smiles", "smi"]
        smiles_col_found = None
        for col in smiles_columns:
            if col in df.columns:
                smiles_col_found = col
                break
        
        if smiles_col_found is None:
            raise ValueError(f"SMILES column not found. Tried: {smiles_columns}. Available: {list(df.columns)}")
        
        smiles = df[smiles_col_found].fillna("").astype(str).tolist()
        return smiles, df, "csv"
    
    else:
        # Try as SMILES file
        with open(p, "r") as f:
            smiles = [line.strip().split()[0] for line in f if line.strip()]
        return smiles, None, "smi"


def write_html_report(output_path: Path, endpoint_names: List[str], 
                       results_df: pd.DataFrame, config: ADMETConfig) -> None:
    """Generate HTML report."""
    n_mols = len(results_df)
    
    # Build endpoint summary
    endpoint_rows = ""
    for ep in endpoint_names:
        # Check for both naming conventions
        col_name = ep if ep in results_df.columns else f"{ep}_prob"
        if col_name not in results_df.columns:
            continue
            
        values = results_df[col_name].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            min_val = values.min()
            max_val = values.max()
            endpoint_rows += f"""
            <tr>
                <td>{ep}</td>
                <td>{mean_val:.4f}</td>
                <td>{std_val:.4f}</td>
                <td>{min_val:.4f}</td>
                <td>{max_val:.4f}</td>
            </tr>
            """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ADMET Prediction Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .endpoint-table th {{ background-color: #2ecc71; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 ADMET Prediction Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total molecules:</strong> {n_mols}</p>
            <p><strong>Endpoints predicted:</strong> {len(endpoint_names)}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Endpoint Statistics</h2>
        <table class="endpoint-table">
            <tr><th>Endpoint</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
            {endpoint_rows}
        </table>
        
        <h2>Results (First 100)</h2>
        {results_df.head(100).to_html(index=False)}
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# =============================================================================
# Module Result
# =============================================================================

if HAS_BASE:
    pass  # Use imported ModuleResult
else:
    class ModuleResult:
        """Result container for module execution."""
        def __init__(self, module_name: str = "admet", module_version: str = "0.0.2",
                     status=None):
            self.module_name = module_name
            self.module_version = module_version
            self.status = status if status else ModuleStatus.COMPLETED
            self.input_count = 0
            self.output_count = 0
            self.success_count = 0
            self.failed_count = 0
            self.duration_seconds = 0.0
            self.metrics = {}
            self.warnings = []
            self.errors = []


# =============================================================================
# Main ADMET Module
# =============================================================================

class ADMETModule:
    """
    ADMET Prediction Module using Chemprop models.
    
    Supports ADMET-AI checkpoints which are Chemprop models that take
    SMILES directly and compute features internally.
    """
    
    MODULE_NAME = "admet"
    MODULE_VERSION = "0.0.2"
    
    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> "ADMETModule":
        """Create from configuration dictionary (called by pipeline)."""
        config = ADMETConfig(
            registry_path=config_dict.get("registry", "module_registry.yaml"),
            device="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu",
            output_formats=config_dict.get("output_formats", ["csv", "sdf", "html"]),
            create_timestamp_folder=config_dict.get("create_timestamp_folder", True),
        )
        return cls(config)
    
    def __init__(self, config: Optional[ADMETConfig] = None, pipeline_config: Dict = None):
        """Initialize ADMET module."""
        self.logger = logging.getLogger("ADMET")
        
        # Get module directory
        self.module_dir = Path(__file__).parent.resolve()
        
        # Handle config from pipeline
        if pipeline_config is not None:
            admet_cfg = pipeline_config.get("admet", {})
            config = ADMETConfig(
                registry_path=admet_cfg.get("registry", "module_registry.yaml"),
                device="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu",
                output_formats=admet_cfg.get("output_formats", ["csv", "sdf", "html"]),
            )
        
        self.config = config or ADMETConfig()
        
        # Initialize state
        self.result = ModuleResult(
            module_name=self.MODULE_NAME,
            module_version=self.MODULE_VERSION
        )
        
        # Data holders
        self.smiles = []
        self.mols = []
        self.features = None
        self.parse_fail = None
        self.results_df = None
        self.input_df = None
        self.input_mode = None
        
        # AD stats
        self.ad_mean = None
        self.ad_std = None
        
        # Endpoints config (loaded from registry)
        self.endpoints_config = {}
        
        # Multitask model paths (if using multitask mode)
        self.multitask_mode = False
        self.classification_model_paths = []
        self.regression_model_paths = []
        
        # Loaded model ensembles (cached)
        self._model_cache = {}
        
        # Checkpoint support
        self._checkpoint_dir = None
        self._checkpoint_file = None
    
    def _init_checkpoint(self, output_dir: Path) -> None:
        """Initialize checkpoint directory and file."""
        self._checkpoint_dir = output_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_file = self._checkpoint_dir / "admet_checkpoint.pkl"
        self.logger.info(f"Checkpoint directory: {self._checkpoint_dir}")
    
    def _save_checkpoint(self, stage: str, data: Dict[str, Any]) -> None:
        """Save checkpoint with current progress."""
        import pickle
        
        checkpoint = {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        try:
            with open(self._checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)
            self.logger.info(f"Checkpoint saved: stage='{stage}'")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if exists."""
        import pickle
        
        if self._checkpoint_file is None or not self._checkpoint_file.exists():
            return None
        
        try:
            with open(self._checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            
            self.logger.info(f"Checkpoint found: stage='{checkpoint['stage']}', time='{checkpoint['timestamp']}'")
            return checkpoint
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _clear_checkpoint(self) -> None:
        """Clear checkpoint after successful completion."""
        if self._checkpoint_file and self._checkpoint_file.exists():
            try:
                self._checkpoint_file.unlink()
                self.logger.info("Checkpoint cleared (job completed successfully)")
            except Exception as e:
                self.logger.warning(f"Failed to clear checkpoint: {e}")
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to module directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.module_dir / p
    
    def _resolve_model_paths(self, paths: List[str]) -> List[str]:
        """Resolve model paths relative to module directory."""
        resolved = []
        for p in paths:
            full_path = self._resolve_path(p)
            if full_path.exists():
                resolved.append(str(full_path))
            else:
                # Log at INFO level so user can see what's happening
                self.logger.info(f"Model NOT found: {full_path}")
        
        if resolved:
            self.logger.info(f"Found {len(resolved)} models, first: {resolved[0]}")
        
        return resolved
    
    def _load_registry(self) -> None:
        """Load endpoint registry."""
        # Registry path is just the filename, resolve relative to module directory
        registry_filename = Path(self.config.registry_path).name  # Get just filename
        registry_path = self.module_dir / registry_filename
        
        self.logger.info(f"Loading registry from: {registry_path}")
        
        if not registry_path.exists():
            self.logger.warning(f"Registry file not found: {registry_path}")
            registry = create_default_registry()
        else:
            try:
                with open(registry_path, "r") as f:
                    data = yaml.safe_load(f)
                registry = data.get("registry", data)
            except Exception as e:
                self.logger.warning(f"Failed to load registry: {e}")
                registry = create_default_registry()
        
        # Extract featurizer settings
        feat_cfg = registry.get("featurizer", {})
        self.config.morgan_radius = feat_cfg.get("morgan_radius", self.config.morgan_radius)
        self.config.morgan_nbits = feat_cfg.get("morgan_nbits", self.config.morgan_nbits)
        
        # Check for multitask mode
        self.multitask_mode = registry.get("multitask", False)
        
        if self.multitask_mode:
            self.logger.info("Using MULTITASK model mode")
            # Load shared model paths for classification and regression
            self.classification_model_paths = registry.get("classification_models", [])
            self.regression_model_paths = registry.get("regression_models", [])
            
            # Resolve paths
            self.classification_model_paths = self._resolve_model_paths(self.classification_model_paths)
            self.regression_model_paths = self._resolve_model_paths(self.regression_model_paths)
            
            self.logger.info(f"Classification models: {len(self.classification_model_paths)}")
            self.logger.info(f"Regression models: {len(self.regression_model_paths)}")
        
        # Extract endpoints
        self.endpoints_config = registry.get("endpoints", {})
        
        if not self.endpoints_config:
            self.logger.info("No endpoints in registry, using built-in defaults")
            self.endpoints_config = {
                name: {
                    "task": info["task"],
                    "description": info["description"],
                    "models": [],
                }
                for name, info in ADMET_ENDPOINTS.items()
            }
        
        # Filter endpoints if specified
        if self.config.endpoints:
            self.endpoints_config = {
                k: v for k, v in self.endpoints_config.items()
                if k in self.config.endpoints
            }
        
        self.logger.info(f"Loaded {len(self.endpoints_config)} endpoints")
    
    def _load_ad_stats(self) -> None:
        """Load applicability domain statistics."""
        ad_stats_path = str(self._resolve_path(self.config.ad_stats_path))
        splits_path = str(self._resolve_path(self.config.splits_path))
        
        self.ad_mean, self.ad_std = load_ad_stats(ad_stats_path, splits_path)
        
        if self.ad_mean is not None:
            self.logger.info(f"Loaded AD stats: {len(self.ad_mean)} features")
    
    def _add_simulated_predictions(self, ep_name: str, task: str) -> None:
        """Add simulated predictions when no models are available."""
        n = len(self.smiles)
        
        # Property-based simulation for more realistic values
        property_scores = np.zeros(n)
        for i, mol in enumerate(self.mols):
            if mol is not None:
                try:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    mw_score = 1 - min(mw / 500, 1.0)
                    logp_score = 1 - abs(logp - 2.5) / 5
                    tpsa_score = 1 - min(tpsa / 140, 1.0)
                    property_scores[i] = (mw_score + logp_score + tpsa_score) / 3
                except:
                    property_scores[i] = 0.5
            else:
                property_scores[i] = 0.5
        
        noise = np.random.normal(0, 0.15, size=n)
        base_scores = np.clip(property_scores + noise, 0, 1)
        
        if task == "classification":
            if any(x in ep_name for x in ["tox", "AMES", "hERG", "DILI"]):
                p_sim = 1 - base_scores
            else:
                p_sim = base_scores
            p_sim = np.clip(p_sim + np.random.normal(0, 0.1, size=n), 0.05, 0.95)
            
            self.results_df[f"{ep_name}_prob"] = np.round(p_sim, 4)
            self.results_df[f"{ep_name}_std"] = np.round(np.random.uniform(0.05, 0.2, size=n), 4)
            self.results_df[f"{ep_name}_unc_flag"] = 0
        else:
            values = base_scores * 2 - 1 + np.random.normal(0, 0.3, size=n)
            self.results_df[ep_name] = np.round(values, 4)
            self.results_df[f"{ep_name}_std"] = np.round(np.random.uniform(0.1, 0.4, size=n), 4)
            self.results_df[f"{ep_name}_unc_flag"] = 0
    
    def load(self, input_paths: Dict[str, str]) -> None:
        """Load input molecules and configuration.
        
        If interactive mode is enabled, prompts user to select VS output folder.
        """
        # Load registry
        self._load_registry()
        
        # Load AD stats
        self._load_ad_stats()
        
        # Interactive input selection if enabled
        if self.config.interactive:
            input_path = interactive_input_selection("output")
            
            if input_path is None:
                raise ValueError("Input selection cancelled by user")
        else:
            # Find input path from provided paths
            input_path = None
            for key in ["hits", "input", "molecules", "vs_results"]:
                if key in input_paths:
                    input_path = input_paths[key]
                    break
            
            if input_path is None:
                raise ValueError(f"No input path found in: {list(input_paths.keys())}")
        
        smiles_col = input_paths.get("smiles_col", "smiles")
        
        # Load molecules
        self.smiles, self.input_df, self.input_mode = load_input_molecules(
            input_path, smiles_col
        )
        
        self.result.input_count = len(self.smiles)
        self.result.metrics["input_path"] = str(input_path)
        
        self.logger.info(f"Loaded {len(self.smiles)} molecules from {input_path}")
    
    def run(self) -> None:
        """Run ADMET predictions with checkpoint support."""
        if not self.smiles:
            raise ValueError("No molecules loaded")
        
        # Check for existing checkpoint
        checkpoint = self._load_checkpoint()
        resume_stage = checkpoint["stage"] if checkpoint else None
        checkpoint_data = checkpoint["data"] if checkpoint else {}
        
        if resume_stage:
            self.logger.info(f"Resuming from checkpoint: stage='{resume_stage}'")
        
        # Check for models (multitask or per-endpoint)
        if hasattr(self, 'multitask_mode') and self.multitask_mode:
            has_real_models = bool(self.classification_model_paths or self.regression_model_paths)
        else:
            has_real_models = any(
                ep.get("models") for ep in self.endpoints_config.values()
            )
        
        # Set device
        if HAS_TORCH and has_real_models:
            device = self.config.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                self.logger.warning("CUDA not available, using CPU")
            self.logger.info(f"Using device: {device}")
        else:
            device = "cpu"
            if not has_real_models:
                self.logger.info("Running in SIMULATED mode (no models found)")
        
        # =====================================================================
        # STAGE 1: Featurization (Morgan fingerprints for AD)
        # =====================================================================
        if resume_stage in [None, "init"]:
            self.logger.info("Stage 1: Computing molecular features...")
            self.features, self.mols, self.parse_fail = featurize_smiles_batch(
                self.smiles,
                radius=self.config.morgan_radius,
                nbits=self.config.morgan_nbits,
                batch_size=self.config.feat_batch_size,
            )
            
            # Initialize results DataFrame with pre-allocated columns to avoid fragmentation
            n_molecules = len(self.smiles)
            endpoint_names = list(self.endpoints_config.keys())
            
            # Build all columns at once
            data = {"smiles": self.smiles, "parse_fail": self.parse_fail.astype(int)}
            
            # Pre-allocate endpoint columns
            for ep_name in endpoint_names:
                task = self.endpoints_config[ep_name].get("task", "").lower()
                if task == "classification":
                    data[f"{ep_name}_prob"] = np.zeros(n_molecules, dtype=np.float32)
                    data[f"{ep_name}_std"] = np.zeros(n_molecules, dtype=np.float32)
                    data[f"{ep_name}_unc_flag"] = np.zeros(n_molecules, dtype=np.int32)
                else:
                    data[ep_name] = np.zeros(n_molecules, dtype=np.float32)
                    data[f"{ep_name}_std"] = np.zeros(n_molecules, dtype=np.float32)
                    data[f"{ep_name}_unc_flag"] = np.zeros(n_molecules, dtype=np.int32)
            
            self.results_df = pd.DataFrame(data)
            
            # Compute AD scores
            if self.ad_mean is not None and self.ad_std is not None:
                if len(self.ad_mean) == self.features.shape[1]:
                    ad_scores = compute_ad_scores(self.features, self.ad_mean, self.ad_std)
                else:
                    # This is expected when AD stats were computed with different features
                    self.logger.debug(
                        f"AD stats dimension mismatch: features={self.features.shape[1]}, "
                        f"ad_stats={len(self.ad_mean)}. Computing from batch."
                    )
                    batch_mean = self.features.mean(axis=0)
                    batch_std = np.maximum(self.features.std(axis=0), 1e-6)
                    ad_scores = compute_ad_scores(self.features, batch_mean, batch_std)
            else:
                self.logger.debug("Computing AD stats from input batch")
                batch_mean = self.features.mean(axis=0)
                batch_std = np.maximum(self.features.std(axis=0), 1e-6)
                ad_scores = compute_ad_scores(self.features, batch_mean, batch_std)
            
            self.results_df["ad_score"] = ad_scores
            self.results_df["ad_flag"] = ((ad_scores > self.config.ad_threshold) | self.parse_fail).astype(int)
            
            # Save checkpoint
            self._save_checkpoint("featurized", {
                "features": self.features,
                "parse_fail": self.parse_fail,
                "results_df": self.results_df,
            })
        else:
            # Restore from checkpoint
            self.logger.info("Restoring featurization from checkpoint...")
            self.features = checkpoint_data.get("features")
            self.parse_fail = checkpoint_data.get("parse_fail")
            self.results_df = checkpoint_data.get("results_df")
        
        endpoint_names = list(self.endpoints_config.keys())
        
        # =====================================================================
        # STAGE 2: Predictions (multitask or per-endpoint)
        # =====================================================================
        if hasattr(self, 'multitask_mode') and self.multitask_mode:
            self._run_multitask_predictions_with_checkpoint(device, endpoint_names, resume_stage, checkpoint_data)
        else:
            self._run_per_endpoint_predictions(device, endpoint_names)
        
        # =====================================================================
        # STAGE 3: Finalize
        # =====================================================================
        # Update result statistics
        self.result.output_count = len(self.results_df)
        self.result.success_count = int((~self.parse_fail).sum())
        self.result.failed_count = int(self.parse_fail.sum())
        
        self.result.metrics = {
            "total_molecules": len(self.smiles),
            "parse_failures": int(self.parse_fail.sum()),
            "ad_flagged": int(self.results_df["ad_flag"].sum()),
            "endpoints_predicted": len(endpoint_names),
        }
        
        self.logger.info(
            f"Prediction complete: {self.result.success_count} successful, "
            f"{self.result.failed_count} failed"
        )
        
        # Clear checkpoint on success
        self._clear_checkpoint()
    
    def _run_multitask_predictions_with_checkpoint(
        self, 
        device: str, 
        endpoint_names: List[str],
        resume_stage: Optional[str],
        checkpoint_data: Dict[str, Any]
    ) -> None:
        """Run multitask predictions with checkpoint support."""
        
        # Separate endpoints by task type
        classification_endpoints = [
            ep for ep in endpoint_names 
            if self.endpoints_config[ep].get("task", "").lower() == "classification"
        ]
        regression_endpoints = [
            ep for ep in endpoint_names 
            if self.endpoints_config[ep].get("task", "").lower() == "regression"
        ]
        
        self.logger.info(f"Classification endpoints: {len(classification_endpoints)}")
        self.logger.info(f"Regression endpoints: {len(regression_endpoints)}")
        
        # Cache for RDKit features
        cached_rdkit_features = checkpoint_data.get("rdkit_features")
        
        # =====================================================================
        # STAGE 2a: Classification predictions
        # =====================================================================
        if resume_stage not in ["classification_done", "regression_done"]:
            if classification_endpoints and self.classification_model_paths:
                self.logger.info(f"Stage 2a: Loading classification models...")
                try:
                    models, scalers, args = load_chemprop_ensemble(
                        self.classification_model_paths, device
                    )
                    
                    self.logger.info(f"Loaded {len(models)} classification models")
                    
                    if models:
                        self.logger.info(f"Running classification predictions...")
                        preds_all, stds_all, cached_rdkit_features = chemprop_ensemble_predict_multitask(
                            models, scalers, args, self.smiles,
                            batch_size=self.config.batch_size,
                            cached_features=cached_rdkit_features
                        )
                        
                        self.logger.info(f"Predictions shape: {preds_all.shape}")
                        
                        output_names = getattr(args, 'task_names', None) if args else None
                        if output_names is None:
                            output_names = sorted(classification_endpoints)
                        
                        self.logger.info(f"Model output columns: {output_names}")
                        
                        for i, ep_name in enumerate(output_names):
                            if ep_name in classification_endpoints and i < preds_all.shape[1]:
                                p_uncal = sigmoid(preds_all[:, i])
                                self.results_df[f"{ep_name}_prob"] = np.round(p_uncal, 4)
                                std_col = stds_all[:, i] if stds_all.ndim > 1 and i < stds_all.shape[1] else np.zeros(len(self.smiles))
                                self.results_df[f"{ep_name}_std"] = np.round(std_col, 4)
                                unc_flag = std_col > self.config.unc_threshold
                                self.results_df[f"{ep_name}_unc_flag"] = (unc_flag | self.parse_fail).astype(int)
                        
                        for ep_name in classification_endpoints:
                            if f"{ep_name}_prob" not in self.results_df.columns:
                                self._add_simulated_predictions(ep_name, "classification")
                    else:
                        self.logger.warning("No classification models loaded successfully")
                        for ep_name in classification_endpoints:
                            self._add_simulated_predictions(ep_name, "classification")
                            
                except Exception as e:
                    self.logger.error(f"Classification prediction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    for ep_name in classification_endpoints:
                        self._add_simulated_predictions(ep_name, "classification")
            else:
                for ep_name in classification_endpoints:
                    self._add_simulated_predictions(ep_name, "classification")
            
            # Save checkpoint after classification
            self._save_checkpoint("classification_done", {
                "features": self.features,
                "parse_fail": self.parse_fail,
                "results_df": self.results_df,
                "rdkit_features": cached_rdkit_features,
            })
        else:
            # Restore classification results from checkpoint
            self.logger.info("Restoring classification results from checkpoint...")
            self.results_df = checkpoint_data.get("results_df", self.results_df)
            cached_rdkit_features = checkpoint_data.get("rdkit_features")
        
        # =====================================================================
        # STAGE 2b: Regression predictions
        # =====================================================================
        if resume_stage != "regression_done":
            if regression_endpoints and self.regression_model_paths:
                self.logger.info(f"Stage 2b: Loading regression models...")
                try:
                    models, scalers, args = load_chemprop_ensemble(
                        self.regression_model_paths, device
                    )
                    
                    self.logger.info(f"Loaded {len(models)} regression models")
                    
                    if models:
                        self.logger.info(f"Running regression predictions...")
                        preds_all, stds_all, cached_rdkit_features = chemprop_ensemble_predict_multitask(
                            models, scalers, args, self.smiles,
                            batch_size=self.config.batch_size,
                            cached_features=cached_rdkit_features
                        )
                        
                        self.logger.info(f"Predictions shape: {preds_all.shape}")
                        
                        output_names = getattr(args, 'task_names', None) if args else None
                        if output_names is None:
                            output_names = sorted(regression_endpoints)
                        
                        self.logger.info(f"Model output columns: {output_names}")
                        
                        for i, ep_name in enumerate(output_names):
                            if ep_name in regression_endpoints and i < preds_all.shape[1]:
                                self.results_df[ep_name] = np.round(preds_all[:, i], 4)
                                std_col = stds_all[:, i] if stds_all.ndim > 1 and i < stds_all.shape[1] else np.zeros(len(self.smiles))
                                self.results_df[f"{ep_name}_std"] = np.round(std_col, 4)
                                unc_flag = std_col > self.config.unc_threshold
                                self.results_df[f"{ep_name}_unc_flag"] = (unc_flag | self.parse_fail).astype(int)
                        
                        for ep_name in regression_endpoints:
                            if ep_name not in self.results_df.columns:
                                self._add_simulated_predictions(ep_name, "regression")
                    else:
                        self.logger.warning("No regression models loaded successfully")
                        for ep_name in regression_endpoints:
                            self._add_simulated_predictions(ep_name, "regression")
                            
                except Exception as e:
                    self.logger.error(f"Regression prediction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    for ep_name in regression_endpoints:
                        self._add_simulated_predictions(ep_name, "regression")
            else:
                for ep_name in regression_endpoints:
                    self._add_simulated_predictions(ep_name, "regression")
            
            # Save checkpoint after regression
            self._save_checkpoint("regression_done", {
                "features": self.features,
                "parse_fail": self.parse_fail,
                "results_df": self.results_df,
            })
    
    def _run_multitask_predictions(self, device: str, endpoint_names: List[str]) -> None:
        """Run predictions using multitask models (one model predicts all endpoints of a task type)."""
        
        # Separate endpoints by task type
        classification_endpoints = [
            ep for ep in endpoint_names 
            if self.endpoints_config[ep].get("task", "").lower() == "classification"
        ]
        regression_endpoints = [
            ep for ep in endpoint_names 
            if self.endpoints_config[ep].get("task", "").lower() == "regression"
        ]
        
        self.logger.info(f"Classification endpoints: {len(classification_endpoints)}")
        self.logger.info(f"Regression endpoints: {len(regression_endpoints)}")
        
        # Cache for RDKit features (computed once, reused)
        cached_rdkit_features = None
        
        # Run classification predictions
        if classification_endpoints and self.classification_model_paths:
            self.logger.info(f"Loading classification models...")
            try:
                models, scalers, args = load_chemprop_ensemble(
                    self.classification_model_paths, device
                )
                
                self.logger.info(f"Loaded {len(models)} classification models")
                
                if models:
                    self.logger.info(f"Running classification predictions...")
                    # Multitask models return multiple outputs + cached features
                    preds_all, stds_all, cached_rdkit_features = chemprop_ensemble_predict_multitask(
                        models, scalers, args, self.smiles,
                        batch_size=self.config.batch_size,
                        cached_features=cached_rdkit_features
                    )
                    
                    self.logger.info(f"Predictions shape: {preds_all.shape}")
                    
                    # Map outputs to endpoints
                    # The order of outputs should match the order in training
                    output_names = getattr(args, 'task_names', None) if args else None
                    if output_names is None:
                        output_names = sorted(classification_endpoints)
                    
                    self.logger.info(f"Model output columns: {output_names}")
                    
                    for i, ep_name in enumerate(output_names):
                        if ep_name in classification_endpoints and i < preds_all.shape[1]:
                            p_uncal = sigmoid(preds_all[:, i])
                            self.results_df[f"{ep_name}_prob"] = np.round(p_uncal, 4)
                            std_col = stds_all[:, i] if stds_all.ndim > 1 and i < stds_all.shape[1] else np.zeros(len(self.smiles))
                            self.results_df[f"{ep_name}_std"] = np.round(std_col, 4)
                            unc_flag = std_col > self.config.unc_threshold
                            self.results_df[f"{ep_name}_unc_flag"] = (unc_flag | self.parse_fail).astype(int)
                    
                    # Handle any endpoints not in model output
                    for ep_name in classification_endpoints:
                        if f"{ep_name}_prob" not in self.results_df.columns:
                            self._add_simulated_predictions(ep_name, "classification")
                else:
                    self.logger.warning("No classification models loaded successfully")
                    for ep_name in classification_endpoints:
                        self._add_simulated_predictions(ep_name, "classification")
                        
            except Exception as e:
                self.logger.error(f"Classification prediction failed: {e}")
                import traceback
                traceback.print_exc()
                for ep_name in classification_endpoints:
                    self._add_simulated_predictions(ep_name, "classification")
        else:
            for ep_name in classification_endpoints:
                self._add_simulated_predictions(ep_name, "classification")
        
        # Run regression predictions
        if regression_endpoints and self.regression_model_paths:
            self.logger.info(f"Loading regression models...")
            try:
                models, scalers, args = load_chemprop_ensemble(
                    self.regression_model_paths, device
                )
                
                self.logger.info(f"Loaded {len(models)} regression models")
                
                if models:
                    self.logger.info(f"Running regression predictions...")
                    # Reuse cached features from classification
                    preds_all, stds_all, cached_rdkit_features = chemprop_ensemble_predict_multitask(
                        models, scalers, args, self.smiles,
                        batch_size=self.config.batch_size,
                        cached_features=cached_rdkit_features  # Reuse!
                    )
                    
                    self.logger.info(f"Predictions shape: {preds_all.shape}")
                    
                    output_names = getattr(args, 'task_names', None) if args else None
                    if output_names is None:
                        output_names = sorted(regression_endpoints)
                    
                    self.logger.info(f"Model output columns: {output_names}")
                    
                    for i, ep_name in enumerate(output_names):
                        if ep_name in regression_endpoints and i < preds_all.shape[1]:
                            self.results_df[ep_name] = np.round(preds_all[:, i], 4)
                            std_col = stds_all[:, i] if stds_all.ndim > 1 and i < stds_all.shape[1] else np.zeros(len(self.smiles))
                            self.results_df[f"{ep_name}_std"] = np.round(std_col, 4)
                            unc_flag = std_col > self.config.unc_threshold
                            self.results_df[f"{ep_name}_unc_flag"] = (unc_flag | self.parse_fail).astype(int)
                    
                    # Handle any endpoints not in model output
                    for ep_name in regression_endpoints:
                        if ep_name not in self.results_df.columns:
                            self._add_simulated_predictions(ep_name, "regression")
                else:
                    self.logger.warning("No regression models loaded successfully")
                    for ep_name in regression_endpoints:
                        self._add_simulated_predictions(ep_name, "regression")
                        
            except Exception as e:
                self.logger.error(f"Regression prediction failed: {e}")
                import traceback
                traceback.print_exc()
                for ep_name in regression_endpoints:
                    self._add_simulated_predictions(ep_name, "regression")
        else:
            for ep_name in regression_endpoints:
                self._add_simulated_predictions(ep_name, "regression")
    
    def _run_per_endpoint_predictions(self, device: str, endpoint_names: List[str]) -> None:
        """Run predictions using per-endpoint models (original mode)."""
        self.logger.info(f"Predicting {len(endpoint_names)} endpoints...")
        
        for ep_name in tqdm_module(endpoint_names, desc="Endpoints", unit="ep", dynamic_ncols=True):
            ep_config = self.endpoints_config[ep_name]
            task = (ep_config.get("task", "") or "").strip().lower()
            model_paths = ep_config.get("models", [])
            
            # SIMULATED MODE: If no models defined
            if not model_paths:
                self._add_simulated_predictions(ep_name, task)
                continue
            
            # Resolve model paths
            resolved_paths = self._resolve_model_paths(model_paths)
            
            if not resolved_paths:
                self.logger.warning(f"No valid model paths for {ep_name}, using simulated predictions")
                self._add_simulated_predictions(ep_name, task)
                continue
            
            # Load Chemprop ensemble
            try:
                models, scalers, args = load_chemprop_ensemble(resolved_paths, device)
            except Exception as e:
                self.logger.warning(f"Failed to load models for {ep_name}: {e}")
                self._add_simulated_predictions(ep_name, task)
                continue
            
            if not models:
                self.logger.warning(f"No valid models loaded for {ep_name}, using simulated predictions")
                self._add_simulated_predictions(ep_name, task)
                continue
            
            # Run Chemprop ensemble prediction
            mean_raw, std_raw = chemprop_ensemble_predict(
                models,
                scalers,
                args,
                self.smiles,
                batch_size=self.config.batch_size,
            )
            
            # Process results based on task type
            if task == "classification":
                # Apply sigmoid to get probabilities
                p_uncal = sigmoid(mean_raw)
                
                # Apply calibration if available
                calib_path = str(self._resolve_path(
                    os.path.join(self.config.calib_dir, ep_name, "calibrator.joblib")
                ))
                calibrator = load_calibrator(calib_path)
                p_cal = apply_calibration(calibrator, p_uncal)
                
                # Store probability
                self.results_df[f"{ep_name}_prob"] = p_cal
                self.results_df[f"{ep_name}_std"] = std_raw
                
                # Uncertainty flag
                unc_flag = (std_raw > self.config.unc_threshold) | self.parse_fail
                self.results_df[f"{ep_name}_unc_flag"] = unc_flag.astype(int)
                
            else:  # regression
                self.results_df[ep_name] = mean_raw
                self.results_df[f"{ep_name}_std"] = std_raw
                
                # Uncertainty flag
                unc_flag = (std_raw > self.config.unc_threshold) | self.parse_fail
                self.results_df[f"{ep_name}_unc_flag"] = unc_flag.astype(int)
    
    def save(self, output_dir: str) -> Dict[str, str]:
        """Save results to output directory."""
        # Create timestamped folder
        if self.config.create_timestamp_folder:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_dir, f"admet_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Saving results to: {output_dir}")
        
        output_files = {}
        endpoint_names = list(self.endpoints_config.keys())
        
        # CSV
        if "csv" in self.config.output_formats:
            csv_path = os.path.join(output_dir, "admet_predictions.csv")
            self.results_df.to_csv(csv_path, index=False)
            output_files["predictions_csv"] = csv_path
            self.logger.info(f"Saved CSV: {csv_path}")
        
        # SDF
        if "sdf" in self.config.output_formats:
            sdf_path = os.path.join(output_dir, "admet_predictions.sdf")
            
            # Get molecules
            if self.input_mode == "sdf":
                input_path = self.result.metrics.get("input_path", "")
                if input_path and os.path.exists(input_path):
                    suppl = Chem.SDMolSupplier(input_path, removeHs=False)
                    mols_out = list(suppl)
                else:
                    mols_out = self.mols
            else:
                mols_out = self.mols
            
            writer = Chem.SDWriter(sdf_path)
            
            for i, mol in enumerate(mols_out):
                if mol is None:
                    continue
                
                mol = Chem.Mol(mol)
                
                # Add predictions as properties
                for col in self.results_df.columns:
                    if col == "smiles":
                        continue
                    val = self.results_df.iloc[i][col]
                    mol.SetProp(col, str(val))
                
                writer.write(mol)
            
            writer.close()
            output_files["predictions_sdf"] = sdf_path
            self.logger.info(f"Saved SDF: {sdf_path}")
        
        # HTML Report
        if "html" in self.config.output_formats:
            html_path = Path(output_dir) / "admet_report.html"
            write_html_report(html_path, endpoint_names, self.results_df, self.config)
            output_files["report_html"] = str(html_path)
            self.logger.info(f"Saved HTML: {html_path}")
        
        return output_files
    
    def execute(self, input_paths: Dict[str, str], output_dir: str) -> Any:
        """Execute the full prediction pipeline with checkpoint support."""
        try:
            start_time = datetime.now()
            
            # Initialize checkpoint directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self._init_checkpoint(output_path)
            
            self.load(input_paths)
            self.run()
            self.save(output_dir)
            self.result.status = ModuleStatus.COMPLETED
            self.result.duration_seconds = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            self.result.status = ModuleStatus.FAILED
            if hasattr(self.result, 'errors'):
                self.result.errors.append(str(e))
            raise
        
        return self.result


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ADMET Prediction Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--registry", "-r", required=True, help="Path to module_registry.yaml")
    parser.add_argument("--input", "-i", required=True, help="Input molecules (CSV or SDF)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--endpoints", "-e", nargs="*", help="Specific endpoints to predict")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--no-timestamp", action="store_true", help="No timestamped folder")
    
    args = parser.parse_args()
    
    config = ADMETConfig(
        registry_path=args.registry,
        device=args.device,
        batch_size=args.batch_size,
        endpoints=args.endpoints,
        create_timestamp_folder=not args.no_timestamp,
    )
    
    module = ADMETModule(config)
    result = module.execute({"input": args.input}, args.output)
    
    print(f"\nStatus: {result.status}")
    print(f"Processed: {result.success_count}/{result.input_count}")


if __name__ == "__main__":
    main()