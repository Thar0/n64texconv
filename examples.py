#!/usr/bin/env python3
#
# Minimal usage examples.
# Not meant to be runnable out of the box, test data not provided.
#

from n64texconv import Image, tex_size, tex_ci_tlut_count
from n64texconv import G_IM_FMT_RGBA, G_IM_FMT_CI, G_IM_FMT_IA, G_IM_FMT_I
from n64texconv import G_IM_SIZ_4b, G_IM_SIZ_8b, G_IM_SIZ_16b, G_IM_SIZ_32b

# Convert from bin

with open("texture.bin", "rb") as infile:
    texture_data = infile.read()

tex = Image.from_bin(texture_data, 32, 32, G_IM_FMT_RGBA, G_IM_SIZ_32b, None)
tex.to_png("texture.png")

with open("ci_texture.bin", "rb") as infile:
    texture_data = infile.read()

with open("ci_pal.bin", "rb") as infile:
    ci_pal_data = infile.read()

pal = Image.palette_from_bin(ci_pal_data, 16)
tex_ci = Image.from_bin(texture_data, 64, 64, G_IM_FMT_CI, G_IM_SIZ_4b, pal)
tex_ci.to_png("texture_ci.png")



# Convert from png

tex = Image.from_png("texture.png", fmt=G_IM_FMT_RGBA, siz=G_IM_SIZ_32b)
tex.to_bin()

#   Since this is saved above with a particular palette, it will match when round-tripped.
#   The library will not generate a new palette if a png is already an indexed png.
tex_ci = Image.from_png("texture_ci.png", fmt=G_IM_FMT_CI, siz=G_IM_SIZ_4b)
tex_ci.to_bin()
tex_ci.pal.to_bin()



# Convert to greyscale

tex.reformat(G_IM_FMT_I, G_IM_SIZ_8b).to_png("texture_greyscale.png")



# Resize

tex.resize(64,64,mode="point").to_png("texture_resized_point.png")
tex.resize(64,64,mode="bilerp").to_png("texture_resized_bilerp.png")



# Quantize to 256 colors with rgba16 palette

tex = Image.from_png("test.png", fmt=G_IM_FMT_CI, siz=G_IM_SIZ_8b)
tex.pal.to_png("test_pal8.png")
tex.to_png("test_ci8.png")



# Quantize to 16 colors with rgba16 palette

tex = tex.reformat(fmt=G_IM_FMT_CI, siz=G_IM_SIZ_4b)
tex.pal.to_png("test_pal4.png")
tex.to_png("test_ci4.png")
