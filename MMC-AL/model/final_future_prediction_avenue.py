"""Compatibility entry point for the shared paper architecture on Avenue."""

from .final_future_prediction_ped2 import convAE_Unet_Sspcab


# Historical evaluation code imported ``convAE`` from this module. Avenue and
# Ped2 now intentionally share the same U-Net + Memory + SSPCAB + PDM model.
convAE = convAE_Unet_Sspcab

__all__ = ['convAE', 'convAE_Unet_Sspcab']
