"""Shared direct-label prompts: identical at training and inference."""

GROUND_CLASSES = (
    "Grd.2S1", "Grd.BM21", "Grd.BM27", "Grd.BM30", "Grd.BMP2",
    "Grd.Bradley", "Grd.BRDM2", "Grd.BTR152", "Grd.BTR60", "Grd.Gaz66",
    "Grd.Himars", "Grd.Humvee", "Grd.Kamaz", "Grd.Kraz255", "Grd.LAV25",
    "Grd.M113", "Grd.M1Abrams", "Grd.Maz537", "Grd.Merkava", "Grd.MTLB",
    "Grd.PT76", "Grd.SU100", "Grd.SU76", "Grd.T34", "Grd.T55", "Grd.T90",
    "Grd.Ural4320", "Grd.Zil131", "Grd.ZSU234", "Grd.ZSU57",
)
DIRECT_SYSTEM = (
    "Classify the Ground military vehicle in the image into exactly one of these classes: "
    + ", ".join(GROUND_CLASSES)
    + ". Return only <answer>Exact_Class_Name</answer>."
)
DIRECT_USER = "Inspect the image and return only the exact Ground class in <answer>...</answer>."
