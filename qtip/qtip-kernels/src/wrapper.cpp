#include <torch/extension.h>


void decompress_matvec_16_9_2_1_256_1_256(
					  torch::Tensor &compressed,
					  torch::Tensor &codebook,
					  torch::Tensor &x,
					  torch::Tensor &out
					  );

void decompress_matvec_16_9_4_1_1024_1_3072(
					  torch::Tensor &compressed,
					  torch::Tensor &codebook,
					  torch::Tensor &x,
					  torch::Tensor &out
					  );

void decompress_matvec_16_9_4_1_8192_1_3072(
					  torch::Tensor &compressed,
					  torch::Tensor &codebook,
					  torch::Tensor &x,
					  torch::Tensor &out
					  );

void decompress_matvec_16_9_4_1_3072_1_8192(
					  torch::Tensor &compressed,
					  torch::Tensor &codebook,
					  torch::Tensor &x,
					  torch::Tensor &out
					  );

void decompress_matvec_16_9_4_1_3072_1_3072(
					  torch::Tensor &compressed,
					  torch::Tensor &codebook,
					  torch::Tensor &x,
					  torch::Tensor &out
					  );

void decompress_matvec_16_9_2_1_53248_1_16384(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_3_1_53248_1_16384(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_4_1_53248_1_16384(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );
void decompress_matvec_16_9_2_1_16384_1_53248(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_3_1_16384_1_53248(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_4_1_16384_1_53248(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_1024_1_16384(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_3_1_1024_1_16384(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_4_1_1024_1_16384(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_16384_1_16384(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_3_1_16384_1_16384(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_4_1_16384_1_16384(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_4096_1_14336(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_3_1_4096_1_14336(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_4_1_4096_1_14336(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_14336_1_4096(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_3_1_14336_1_4096(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_4_1_14336_1_4096(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_1024_1_4096(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_3_1_1024_1_4096(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_4_1_1024_1_4096(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_4096_1_4096(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_4096_1_11008(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_2_1_11008_1_4096(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_2_1_12288_1_4096(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_2_1_22016_1_4096(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_2_1_8192_1_8192(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_2_1_10240_1_8192(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_3_1_10240_1_8192(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_4_1_10240_1_8192(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_2_1_57344_1_8192(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_3_1_57344_1_8192(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_4_1_57344_1_8192(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_2_1_8192_1_1024(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_8192_1_28672(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_2_1_28672_1_8192(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_2_1_1024_1_8192(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );



void decompress_matvec_16_9_3_1_256_1_256(
					  torch::Tensor &compressed,
					  torch::Tensor &codebook,
					  torch::Tensor &x,
					  torch::Tensor &out
					  );


void decompress_matvec_16_9_3_1_4096_1_4096(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_3_1_4096_1_11008(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_3_1_11008_1_4096(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_3_1_12288_1_4096(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_3_1_22016_1_4096(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );



void decompress_matvec_16_9_3_1_8192_1_8192(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_3_1_8192_1_1024(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_3_1_8192_1_28672(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_3_1_28672_1_8192(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_3_1_1024_1_8192(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );



void decompress_matvec_16_9_4_1_256_1_256(
					  torch::Tensor &compressed,
					  torch::Tensor &codebook,
					  torch::Tensor &x,
					  torch::Tensor &out
					  );


void decompress_matvec_16_9_4_1_4096_1_4096(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_4_1_4096_1_11008(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_4_1_11008_1_4096(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );



void decompress_matvec_16_9_4_1_12288_1_4096(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_4_1_22016_1_4096(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );



void decompress_matvec_16_9_4_1_8192_1_8192(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_4_1_8192_1_1024(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_4_1_8192_1_28672(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );


void decompress_matvec_16_9_4_1_28672_1_8192(
					     torch::Tensor &compressed,
					     torch::Tensor &codebook,
					     torch::Tensor &x,
					     torch::Tensor &out
					     );

void decompress_matvec_16_9_4_1_1024_1_8192(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_2_1_5120_1_5120(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_3_1_5120_1_5120(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_4_1_5120_1_5120(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_5120_1_13824(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_3_1_5120_1_13824(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_4_1_5120_1_13824(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


void decompress_matvec_16_9_2_1_13824_1_5120(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_3_1_13824_1_5120(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );

void decompress_matvec_16_9_4_1_13824_1_5120(
					    torch::Tensor &compressed,
					    torch::Tensor &codebook,
					    torch::Tensor &x,
					    torch::Tensor &out
					    );


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decompress_matvec_16_9_2_1_256_1_256", &decompress_matvec_16_9_2_1_256_1_256, "decompress_matvec_16_9_2_1_256_1_256");

  m.def("decompress_matvec_16_9_4_1_1024_1_3072", &decompress_matvec_16_9_4_1_1024_1_3072, "decompress_matvec_16_9_4_1_1024_1_3072");
  m.def("decompress_matvec_16_9_4_1_8192_1_3072", &decompress_matvec_16_9_4_1_8192_1_3072, "decompress_matvec_16_9_4_1_8192_1_3072");
  m.def("decompress_matvec_16_9_4_1_3072_1_8192", &decompress_matvec_16_9_4_1_3072_1_8192, "decompress_matvec_16_9_4_1_3072_1_8192");
  m.def("decompress_matvec_16_9_4_1_3072_1_3072", &decompress_matvec_16_9_4_1_3072_1_3072, "decompress_matvec_16_9_4_1_3072_1_3072");

  m.def("decompress_matvec_16_9_2_1_53248_1_16384", &decompress_matvec_16_9_2_1_53248_1_16384, "decompress_matvec_16_9_2_1_53248_1_16384");
  m.def("decompress_matvec_16_9_3_1_53248_1_16384", &decompress_matvec_16_9_3_1_53248_1_16384, "decompress_matvec_16_9_3_1_53248_1_16384");
  m.def("decompress_matvec_16_9_4_1_53248_1_16384", &decompress_matvec_16_9_4_1_53248_1_16384, "decompress_matvec_16_9_4_1_53248_1_16384");

  m.def("decompress_matvec_16_9_2_1_16384_1_53248", &decompress_matvec_16_9_2_1_16384_1_53248, "decompress_matvec_16_9_2_1_16384_1_53248");
  m.def("decompress_matvec_16_9_3_1_16384_1_53248", &decompress_matvec_16_9_3_1_16384_1_53248, "decompress_matvec_16_9_3_1_16384_1_53248");
  m.def("decompress_matvec_16_9_4_1_16384_1_53248", &decompress_matvec_16_9_4_1_16384_1_53248, "decompress_matvec_16_9_4_1_16384_1_53248");

  m.def("decompress_matvec_16_9_2_1_1024_1_16384", &decompress_matvec_16_9_2_1_1024_1_16384, "decompress_matvec_16_9_2_1_1024_1_16384");
  m.def("decompress_matvec_16_9_3_1_1024_1_16384", &decompress_matvec_16_9_3_1_1024_1_16384, "decompress_matvec_16_9_3_1_1024_1_16384");
  m.def("decompress_matvec_16_9_4_1_1024_1_16384", &decompress_matvec_16_9_4_1_1024_1_16384, "decompress_matvec_16_9_4_1_1024_1_16384");
  
  m.def("decompress_matvec_16_9_2_1_16384_1_16384", &decompress_matvec_16_9_2_1_16384_1_16384, "decompress_matvec_16_9_2_1_16384_1_16384");
  m.def("decompress_matvec_16_9_3_1_16384_1_16384", &decompress_matvec_16_9_3_1_16384_1_16384, "decompress_matvec_16_9_3_1_16384_1_16384");
  m.def("decompress_matvec_16_9_4_1_16384_1_16384", &decompress_matvec_16_9_4_1_16384_1_16384, "decompress_matvec_16_9_4_1_16384_1_16384");
  
  m.def("decompress_matvec_16_9_2_1_4096_1_14336", &decompress_matvec_16_9_2_1_4096_1_14336, "decompress_matvec_16_9_2_1_4096_1_14336");
  m.def("decompress_matvec_16_9_3_1_4096_1_14336", &decompress_matvec_16_9_3_1_4096_1_14336, "decompress_matvec_16_9_3_1_4096_1_14336");
  m.def("decompress_matvec_16_9_4_1_4096_1_14336", &decompress_matvec_16_9_4_1_4096_1_14336, "decompress_matvec_16_9_4_1_4096_1_14336");

  m.def("decompress_matvec_16_9_2_1_14336_1_4096", &decompress_matvec_16_9_2_1_14336_1_4096, "decompress_matvec_16_9_2_1_14336_1_4096");
  m.def("decompress_matvec_16_9_3_1_14336_1_4096", &decompress_matvec_16_9_3_1_14336_1_4096, "decompress_matvec_16_9_3_1_14336_1_4096");
  m.def("decompress_matvec_16_9_4_1_14336_1_4096", &decompress_matvec_16_9_4_1_14336_1_4096, "decompress_matvec_16_9_4_1_14336_1_4096");

  m.def("decompress_matvec_16_9_2_1_1024_1_4096", &decompress_matvec_16_9_2_1_1024_1_4096, "decompress_matvec_16_9_2_1_1024_1_4096");
  m.def("decompress_matvec_16_9_3_1_1024_1_4096", &decompress_matvec_16_9_3_1_1024_1_4096, "decompress_matvec_16_9_3_1_1024_1_4096");
  m.def("decompress_matvec_16_9_4_1_1024_1_4096", &decompress_matvec_16_9_4_1_1024_1_4096, "decompress_matvec_16_9_4_1_1024_1_4096");
  m.def("decompress_matvec_16_9_2_1_4096_1_4096", &decompress_matvec_16_9_2_1_4096_1_4096, "decompress_matvec_16_9_2_1_4096_1_4096");
  m.def("decompress_matvec_16_9_2_1_4096_1_11008", &decompress_matvec_16_9_2_1_4096_1_11008, "decompress_matvec_16_9_2_1_4096_1_11008");
  m.def("decompress_matvec_16_9_2_1_11008_1_4096", &decompress_matvec_16_9_2_1_11008_1_4096, "decompress_matvec_16_9_2_1_11008_1_4096");
  m.def("decompress_matvec_16_9_2_1_12288_1_4096", &decompress_matvec_16_9_2_1_12288_1_4096, "decompress_matvec_16_9_2_1_12288_1_4096");
  m.def("decompress_matvec_16_9_2_1_22016_1_4096", &decompress_matvec_16_9_2_1_22016_1_4096, "decompress_matvec_16_9_2_1_22016_1_4096");
  m.def("decompress_matvec_16_9_2_1_8192_1_8192", &decompress_matvec_16_9_2_1_8192_1_8192, "decompress_matvec_16_9_2_1_8192_1_8192");
  m.def("decompress_matvec_16_9_2_1_10240_1_8192", &decompress_matvec_16_9_2_1_10240_1_8192, "decompress_matvec_16_9_2_1_10240_1_8192");
  m.def("decompress_matvec_16_9_2_1_57344_1_8192", &decompress_matvec_16_9_2_1_57344_1_8192, "decompress_matvec_16_9_2_1_57344_1_8192");
  m.def("decompress_matvec_16_9_3_1_57344_1_8192", &decompress_matvec_16_9_3_1_57344_1_8192, "decompress_matvec_16_9_3_1_57344_1_8192");
  m.def("decompress_matvec_16_9_4_1_57344_1_8192", &decompress_matvec_16_9_4_1_57344_1_8192, "decompress_matvec_16_9_4_1_57344_1_8192");
  m.def("decompress_matvec_16_9_2_1_8192_1_1024", &decompress_matvec_16_9_2_1_8192_1_1024, "decompress_matvec_16_9_2_1_8192_1_1024");
  m.def("decompress_matvec_16_9_2_1_8192_1_28672", &decompress_matvec_16_9_2_1_8192_1_28672, "decompress_matvec_16_9_2_1_8192_1_28672");
  m.def("decompress_matvec_16_9_2_1_28672_1_8192", &decompress_matvec_16_9_2_1_28672_1_8192, "decompress_matvec_16_9_2_1_28672_1_8192");
  m.def("decompress_matvec_16_9_2_1_1024_1_8192", &decompress_matvec_16_9_2_1_1024_1_8192, "decompress_matvec_16_9_2_1_1024_1_8192");

  m.def("decompress_matvec_16_9_3_1_256_1_256", &decompress_matvec_16_9_3_1_256_1_256, "decompress_matvec_16_9_3_1_256_1_256");
  m.def("decompress_matvec_16_9_3_1_4096_1_4096", &decompress_matvec_16_9_3_1_4096_1_4096, "decompress_matvec_16_9_3_1_4096_1_4096");
  m.def("decompress_matvec_16_9_3_1_4096_1_11008", &decompress_matvec_16_9_3_1_4096_1_11008, "decompress_matvec_16_9_3_1_4096_1_11008");
  m.def("decompress_matvec_16_9_3_1_11008_1_4096", &decompress_matvec_16_9_3_1_11008_1_4096, "decompress_matvec_16_9_3_1_11008_1_4096");
  m.def("decompress_matvec_16_9_3_1_12288_1_4096", &decompress_matvec_16_9_3_1_12288_1_4096, "decompress_matvec_16_9_3_1_12288_1_4096");
  m.def("decompress_matvec_16_9_3_1_22016_1_4096", &decompress_matvec_16_9_3_1_22016_1_4096, "decompress_matvec_16_9_3_1_22016_1_4096");
  m.def("decompress_matvec_16_9_3_1_8192_1_8192", &decompress_matvec_16_9_3_1_8192_1_8192, "decompress_matvec_16_9_3_1_8192_1_8192");
  m.def("decompress_matvec_16_9_3_1_10240_1_8192", &decompress_matvec_16_9_3_1_10240_1_8192, "decompress_matvec_16_9_3_1_10240_1_8192");
  m.def("decompress_matvec_16_9_3_1_8192_1_1024", &decompress_matvec_16_9_3_1_8192_1_1024, "decompress_matvec_16_9_3_1_8192_1_1024");
  m.def("decompress_matvec_16_9_3_1_8192_1_28672", &decompress_matvec_16_9_3_1_8192_1_28672, "decompress_matvec_16_9_3_1_8192_1_28672");
  m.def("decompress_matvec_16_9_3_1_28672_1_8192", &decompress_matvec_16_9_3_1_28672_1_8192, "decompress_matvec_16_9_3_1_28672_1_8192");
  m.def("decompress_matvec_16_9_3_1_1024_1_8192", &decompress_matvec_16_9_3_1_1024_1_8192, "decompress_matvec_16_9_3_1_1024_1_8192");

  m.def("decompress_matvec_16_9_4_1_256_1_256", &decompress_matvec_16_9_4_1_256_1_256, "decompress_matvec_16_9_4_1_256_1_256");
  m.def("decompress_matvec_16_9_4_1_4096_1_4096", &decompress_matvec_16_9_4_1_4096_1_4096, "decompress_matvec_16_9_4_1_4096_1_4096");
  m.def("decompress_matvec_16_9_4_1_4096_1_11008", &decompress_matvec_16_9_4_1_4096_1_11008, "decompress_matvec_16_9_4_1_4096_1_11008");
  m.def("decompress_matvec_16_9_4_1_11008_1_4096", &decompress_matvec_16_9_4_1_11008_1_4096, "decompress_matvec_16_9_4_1_11008_1_4096");
  m.def("decompress_matvec_16_9_4_1_12288_1_4096", &decompress_matvec_16_9_4_1_12288_1_4096, "decompress_matvec_16_9_4_1_12288_1_4096");
  m.def("decompress_matvec_16_9_4_1_22016_1_4096", &decompress_matvec_16_9_4_1_22016_1_4096, "decompress_matvec_16_9_4_1_22016_1_4096");
  m.def("decompress_matvec_16_9_4_1_8192_1_8192", &decompress_matvec_16_9_4_1_8192_1_8192, "decompress_matvec_16_9_4_1_8192_1_8192");
  m.def("decompress_matvec_16_9_4_1_10240_1_8192", &decompress_matvec_16_9_4_1_10240_1_8192, "decompress_matvec_16_9_4_1_10240_1_8192");
  m.def("decompress_matvec_16_9_4_1_8192_1_1024", &decompress_matvec_16_9_4_1_8192_1_1024, "decompress_matvec_16_9_4_1_8192_1_1024");
  m.def("decompress_matvec_16_9_4_1_8192_1_28672", &decompress_matvec_16_9_4_1_8192_1_28672, "decompress_matvec_16_9_4_1_8192_1_28672");
  m.def("decompress_matvec_16_9_4_1_28672_1_8192", &decompress_matvec_16_9_4_1_28672_1_8192, "decompress_matvec_16_9_4_1_28672_1_8192");
  m.def("decompress_matvec_16_9_4_1_1024_1_8192", &decompress_matvec_16_9_4_1_1024_1_8192, "decompress_matvec_16_9_4_1_1024_1_8192");

  m.def("decompress_matvec_16_9_2_1_5120_1_5120", &decompress_matvec_16_9_2_1_5120_1_5120, "decompress_matvec_16_9_2_1_5120_1_5120");
  m.def("decompress_matvec_16_9_3_1_5120_1_5120", &decompress_matvec_16_9_3_1_5120_1_5120, "decompress_matvec_16_9_3_1_5120_1_5120");
  m.def("decompress_matvec_16_9_4_1_5120_1_5120", &decompress_matvec_16_9_4_1_5120_1_5120, "decompress_matvec_16_9_4_1_5120_1_5120");

  m.def("decompress_matvec_16_9_2_1_5120_1_13824", &decompress_matvec_16_9_2_1_5120_1_13824, "decompress_matvec_16_9_2_1_5120_1_13824");	
  m.def("decompress_matvec_16_9_3_1_5120_1_13824", &decompress_matvec_16_9_3_1_5120_1_13824, "decompress_matvec_16_9_3_1_5120_1_13824");	
  m.def("decompress_matvec_16_9_4_1_5120_1_13824", &decompress_matvec_16_9_4_1_5120_1_13824, "decompress_matvec_16_9_4_1_5120_1_13824");	
  
  m.def("decompress_matvec_16_9_2_1_13824_1_5120", &decompress_matvec_16_9_2_1_13824_1_5120, "decompress_matvec_16_9_2_1_13824_1_5120");	
  m.def("decompress_matvec_16_9_3_1_13824_1_5120", &decompress_matvec_16_9_3_1_13824_1_5120, "decompress_matvec_16_9_3_1_13824_1_5120");	
  m.def("decompress_matvec_16_9_4_1_13824_1_5120", &decompress_matvec_16_9_4_1_13824_1_5120, "decompress_matvec_16_9_4_1_13824_1_5120");	
  
}
