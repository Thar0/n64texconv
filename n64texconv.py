# MIT License
# 
# Copyright (c) 2022 Tharo
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Conversion between various N64 texture formats and png.
# 
# Requires additional libraries:
#  - pypng
#  - libimagequant

# TODO ensure user-provided palette and palette sharing works as intended

import struct

import png
import libimagequant

G_IM_FMT_RGBA = 0
G_IM_FMT_YUV  = 1   # UNSUPPORTED at this time
G_IM_FMT_CI   = 2
G_IM_FMT_IA   = 3
G_IM_FMT_I    = 4

G_IM_SIZ_4b  = 0
G_IM_SIZ_8b  = 1
G_IM_SIZ_16b = 2
G_IM_SIZ_32b = 3

def exception_if_not(cond, type, msg):
    if not (cond):
        raise type(msg)

def G_SIZ_BYTES(siz):
    """
    Texel size in bytes
    """
    return (0.5, 1, 2, 4)[siz]

def FMTSIZ(fmt, siz):
    return (fmt << 2) | siz

def tex_size(width, height, siz):
    """
    Texture size in bytes with dimensions `width`,`height` with texel size `siz`
    """
    return int(width * height * G_SIZ_BYTES(siz))

def tex_ci_tlut_count(data, siz):
    """
    Returns the largest index + 1 used by the CI texture `data` with texel size `siz`.
    The texel size should be 4b or 8b, the only valid CI sizes.
    """
    exception_if_not(siz in [G_IM_SIZ_4b, G_IM_SIZ_8b], Exception, "Invalid size for CI texture")

    if siz == G_IM_SIZ_4b:
        max_v = -1
        for b in data:
            b1 = (b >> 4) & 0xF
            b2 = (b >> 0) & 0xF
            max_v = max(max_v, b1, b2)
        return max_v + 1
    else:
        return max(data) + 1

def pal_search(px, pal_texels):
    # check if px is in pal
    for i,p in enumerate(pal_texels):
        if p == px:
            return i
    # not found
    return -1

def texel_greyscale(r, g, b, a):
    return int(round(r * 0.299 + g * 0.587 + b * 0.114))

def texel_greyscale_alpha(r, g, b, a):
    return texel_greyscale(r,g,b,a) * a

def texels_to_bytes(texels):
    img = bytearray()
    for txl in texels:
        img.extend(txl)
    return bytes(img)

def color_quant(texels, width, height, num, dither_lvl=0.5): # TODO adjust default
    # check if the image already consists of `num` colors
    unique_texels = set(texels) # TODO ensure this is deterministic
    if len(unique_texels) <= num:
        # already has sufficiently few colors
        quant_colors = list(unique_texels)
        quant_colors.extend([(0,0,0,0)] * (num - len(unique_texels)))
        assert len(quant_colors) == num

        # map texels to quant_colors index
        quant_idxs = [pal_search(px, quant_colors) for px in texels]
        assert all([idx >= 0 and idx < num for idx in quant_idxs])
    else:
        # quantize with libimagequant
        attr = libimagequant.Attr()
        attr.max_colors = num
        liq_img = attr.create_rgba(texels_to_bytes(texels), width, height, 0.0)
        result = liq_img.quantize(attr)
        result.dithering_level = dither_lvl
        # receive results, retrieve palette after remapping for higher quality
        quant_idxs = result.remap_image(liq_img)
        quant_colors = result.get_palette()
        # check palette meets specification
        assert len(quant_colors) == num , "[INTERNAL ERROR] libimagequant produced different-sized palette"
        # convert palette texel data
        quant_colors = [(color.r, color.g, color.b, color.a) for color in quant_colors]

    # generate palette and return mapping + palette
    pal_dim = 4 if num <= 16 else 16
    # TODO generate IA palettes?
    pal = Image.new(pal_dim, pal_dim, G_IM_FMT_RGBA, G_IM_SIZ_16b, None)
    pal.texels = quant_colors
    return quant_idxs, pal

def png_to_data(file):
    png_reader = png.Reader(filename=file)

    # Try and read palette data, which may not exist
    png_reader.preamble()
    try:
        palette_data = png_reader.palette()
    except png.FormatError:
        palette_data = None

    # Read png data
    if palette_data is not None:
        # read color indices as-is, needed for matching existing CI textures
        exception_if_not(len(palette_data) <= 256, Exception,
                         "Palette in indexed png is too large, maximum palette size is 256 colors")
        png_data = png_reader.read()
    else:
        # convert data to RGBA32
        png_data = png_reader.asRGBA8()
        # for non-indexed pngs, the image is quantized

    return png_data, palette_data

class Image:
    @staticmethod
    def new(width, height, fmt, siz, pal):
        fmtsiz_map = {
            FMTSIZ(G_IM_FMT_RGBA, G_IM_SIZ_32b) : RGBA32,
            FMTSIZ(G_IM_FMT_RGBA, G_IM_SIZ_16b) : RGBA16,
            FMTSIZ(G_IM_FMT_IA,   G_IM_SIZ_16b) : IA16,
            FMTSIZ(G_IM_FMT_IA,   G_IM_SIZ_8b ) : IA8,
            FMTSIZ(G_IM_FMT_IA,   G_IM_SIZ_4b ) : IA4,
            FMTSIZ(G_IM_FMT_I,    G_IM_SIZ_8b ) : I8,
            FMTSIZ(G_IM_FMT_I,    G_IM_SIZ_4b ) : I4,
            FMTSIZ(G_IM_FMT_CI,   G_IM_SIZ_8b ) : CI8,
            FMTSIZ(G_IM_FMT_CI,   G_IM_SIZ_4b ) : CI4,
        }
        exception_if_not(FMTSIZ(fmt, siz) in fmtsiz_map, ValueError, "Invalid texture format/size combination")
        return (fmtsiz_map[FMTSIZ(fmt, siz)])(width, height, pal)

    @staticmethod
    def png_to_ci_shared_pal(files, siz=G_IM_SIZ_8b, pal_fmt=G_IM_FMT_RGBA):
        """
        Converts multiple pngs to CI, generating a single shared palette
        """
        # TODO implement
        raise NotImplementedError()

    @staticmethod
    def from_png(file, fmt=None, siz=None, pal_fmt=G_IM_FMT_RGBA):
        """
        Reads in the png image `file`.

        For all choices of `fmt`, the png data is converted to a common RGBA32 format and stored that way, but
        may be subject to various edits described below.

         - If `fmt` is None (default), the best format and texel size are determined from the RGBA32 data, which is
           then processed accordingly (described below).

         - If `fmt` is RGBA, the RGBA32 texels are left as-is, with possible reduction if `siz` is 16-bit.

         - If `fmt` is greyscale (I or IA), the RGBA32 texels are converted to greyscale. If the texels were already
           greyscale this operation will leave them unaffected.

         - If `fmt` is CI and the png is an indexed png, the palette and indices in the png are saved in addition
           to the texels themselves. This is required for a roundtrip conversion of bin -> png -> file -> png -> bin
           to match.
         - If `fmt` is CI and the png is not indexed, the image is quantized and a palette is generated automatically.
        """

        # read png data
        png_data, palette_data = png_to_data(file)

        # create palette if png is indexed
        pal = None
        if palette_data is not None:
            dim = 4 if len(palette_data) <= 16 else 16
            palette_data.extend([(0,0,0,0)] * (dim * dim - len(palette_data)))
            pal = Image.new(dim, dim, pal_fmt, G_IM_SIZ_16b, None)
            pal.texels = palette_data

        print(palette_data)

        # create image
        width = png_data[0]
        height = png_data[1]
        img = Image.new(width, height,
                        fmt if fmt is not None else G_IM_FMT_RGBA,
                        siz if siz is not None else G_IM_SIZ_32b,
                        pal)

        if palette_data is not None:
            img.color_indices = []
            for row in png_data[2]:
                img.color_indices.extend(row)

            # unpack into texels as well for possible conversion to other formats
            img.texels = [img.unpack(t) for t in img.color_indices]
        else:
            img.texels = []
            for row in png_data[2]:
                if fmt == G_IM_FMT_I:
                    # alpha channel fixup for intensity images, TODO move to reformat?
                    img.texels.extend(
                        [(row[i + 0], row[i + 1], row[i + 2], row[i + 2] if row[i + 3] == 0xFF else row[i + 3])
                            for i in range(0, len(row), 4)])
                else:
                    img.texels.extend(
                        [(row[i + 0], row[i + 1], row[i + 2], row[i + 3])
                            for i in range(0, len(row), 4)])

        # reformat the input data, this involves converting to greyscale if I or IA and quantizing the image
        # if the png was not already indexed
        return img.reformat(fmt, siz)

    @staticmethod
    def palette_from_bin(data, count, fmt=G_IM_FMT_RGBA):
        """
        Reads a palette from binary data.
        """
        exception_if_not(count <= 256, ValueError, "Maximum palette size is 256 colors")

        dim = 16 if count > 16 else 4
        tlut_square_count = dim * dim # 256 or 16
        tlut_target_bytes = int(G_SIZ_BYTES(G_IM_SIZ_16b) * tlut_square_count)

        # can this be avoided (probably not)
        data.extend([0] * (tlut_target_bytes - len(data)))

        assert len(data) == tlut_target_bytes, \
               f"[INTERNAL ERROR] Palette target size did not match actual size {tlut_target_bytes} {len(data)}"

        # create
        return Image.from_bin(data, dim, dim, fmt, G_IM_SIZ_16b, None)

    @staticmethod
    def from_bin(data, width, height, fmt, siz, pal, preswapped=False):
        """
        Reads a texture from binary data.
        """
        # TODO implement dxt=0 texture word swapping

        exception_if_not(fmt != G_IM_FMT_CI or pal is not None, Exception,
                         "Reading G_IM_FMT_CI from binary must have a palette supplied")

        # read individual texels from binary format, no conversion is done yet as we do not know
        # if we are dealing with indices or colors just yet

        # total texture size in bytes
        n_bytes = tex_size(width, height, siz)
        # we had better have enough data to read
        exception_if_not(len(data) >= n_bytes, Exception,
                         "Not enough data supplied for this texture width-height-siz combination")

        if siz == G_IM_SIZ_4b:
            # gross 4-bpp unpacking
            texels = []
            for i in range(n_bytes):
                texels.append((data[i] >> 4) & 0xF)
                texels.append((data[i] >> 0) & 0xF)
        else:
            # other formats are multiplies of a byte, struct works for this
            struct_fmt = ">" + (None, "B", "H", "I")[siz]
            texels = [i[0] for i in struct.iter_unpack(struct_fmt, data[:n_bytes])]

        # create new image
        img = Image.new(width, height, fmt, siz, pal)

        # Needed to preserve matching in existing CI textures. The color indices cannot be rebuilt
        # from scratch as some palettes contain duplicate colors, so there is no way to know which
        # index should be used without creating the CI texture exactly as original tools did.
        if fmt == G_IM_FMT_CI:
            img.color_indices = texels

        # Unpack image data from N64 formats to RGBA32
        img.texels = [img.unpack(t) for t in texels]
        return img

    def __init__(self, width, height, pal=None):
        self.width = width
        self.height = height
        self.pal = pal
        self.fmt = type(self).FMT
        self.siz = type(self).SIZ

        self.texels = None
        self.color_indices = None

    def best_fmt_siz(self):
        """
        Using only RGBA32 image data, determine the best fmt+siz combination to represent it.
        """
        # TODO implement
        best_fmt = self.fmt
        best_siz = self.siz

        return best_fmt, best_siz

    def resize(self, new_w, new_h, mode="point"):
        exception_if_not(mode in ["point", "bilerp"], ValueError, "Invalid resize mode")

        old_w = self.width
        old_h = self.height

        new_texels = [(0,0,0,0)] * new_w * new_h
        old_texels = self.texels

        if mode == "point":
            for new_y in range(new_h):
                old_y = min(max(int(((new_y + 0.5) * (old_h / new_h))), 0), old_h - 1)

                for new_x in range(new_w):
                    old_x = min(max(int(((new_x + 0.5) * (old_w / new_w))), 0), old_w - 1)

                    new_texels[new_x + new_y * new_w] = old_texels[old_x + old_y * old_w]

        elif mode == "bilerp":
            # naive bilinear filtering with 3 points like N64, probably not accurate however

            for new_y in range(new_h):
                old_y1 = min(max(int(((new_y + 0) * (old_h / new_h))), 0), old_h - 1)
                old_y2 = min(max(int(((new_y + 0) * (old_h / new_h))), 0), old_h - 1)
                old_y3 = min(max(int(((new_y + 1) * (old_h / new_h))), 0), old_h - 1)

                for new_x in range(new_w):
                    old_x1 = min(max(int(((new_x + 0) * (old_w / new_w))), 0), old_w - 1)
                    old_x2 = min(max(int(((new_x + 1) * (old_w / new_w))), 0), old_w - 1)
                    old_x3 = min(max(int(((new_x + 1) * (old_w / new_w))), 0), old_w - 1)

                    samp1 = old_texels[old_x1 + old_y1 * old_w]
                    samp2 = old_texels[old_x2 + old_y2 * old_w]
                    samp3 = old_texels[old_x3 + old_y3 * old_w]
                    # TODO these should probably be weighted based on distance to center pixel?
                    new_texels[new_x + new_y * new_w] = \
                        [int(round((s1 + s2 + s3) / 3)) for s1,s2,s3 in zip(samp1,samp2,samp3)]

        img = Image.new(new_w, new_h, self.fmt, self.siz, None)
        img.texels = new_texels
        return img.reformat(self.fmt, self.siz)

    def reformat(self, fmt=None, siz=None):
        """
        Given an existing image, create a new image with the new format and size. This operation is very often a lossy
        process, do not expect matching round-trips through reformat; quality is often lost, sometimes severely.
        This process is typically for converting truecolor png to the optimal N64 format if the format was left
        unspecified when reading in the png, in a matching context the format should always be explicitly specified
        and no reformatting done.
        """
        if fmt == None and siz == None:
            # Determine best format if not specified
            fmt, siz = self.best_fmt_siz()

        # New format
        img = Image.new(self.width, self.height, fmt, siz, None)
        img.texels = self.texels

        # Format fixups
        if fmt in [G_IM_FMT_I, G_IM_FMT_IA]:
            # Convert texel data to greyscale
            grey_c = lambda r,g,b,a : texel_greyscale(r,g,b,a)
            grey_a = grey_c if img.fmt == G_IM_FMT_I else (lambda r,g,b,a : a) # either replicate rgb or preserve alpha

            img.texels = [(grey_c(*txl), grey_c(*txl), grey_c(*txl), grey_a(*txl)) for txl in img.texels]
        elif fmt == G_IM_FMT_CI:
            # Convert texel data to CI and generate palette if necessary
            img.color_indices = self.color_indices
            img.pal = self.pal

            # If we have color indices we'd better have a palette
            assert img.color_indices is None or img.pal is not None

            if self.siz != siz:
                img.color_indices = None
                img.pal = None

            if img.color_indices is None:
                # Input data is not already indexed, generate
                # TODO add some method for the user to provide a palette
                n_colors = 16 if (img.siz == G_IM_SIZ_4b) else 256
                img.color_indices, img.pal = color_quant(img.texels, img.width, img.height, n_colors)
                img.texels = [img.unpack(i) for i in img.color_indices]

        img.texels = [img.unpack(img.pack(p[0] & 0xFF, p[1] & 0xFF, p[2] & 0xFF, p[3] & 0xFF)) for p in img.texels]

        assert img.fmt != G_IM_FMT_CI or img.pal is not None

        return img

    def png_extension(self):
        """
        Create a png file extension that reflects the intended texture format
        """
        extensions = {
            FMTSIZ(G_IM_FMT_RGBA, G_IM_SIZ_16b) : "rgba16",
            FMTSIZ(G_IM_FMT_RGBA, G_IM_SIZ_32b) : "rgba32",
            FMTSIZ(G_IM_FMT_I,    G_IM_SIZ_4b)  : "i4"    ,
            FMTSIZ(G_IM_FMT_I,    G_IM_SIZ_8b)  : "i8"    ,
            FMTSIZ(G_IM_FMT_IA,   G_IM_SIZ_4b)  : "ia4"   ,
            FMTSIZ(G_IM_FMT_IA,   G_IM_SIZ_8b)  : "ia8"   ,
            FMTSIZ(G_IM_FMT_IA,   G_IM_SIZ_16b) : "ia16"  ,
            FMTSIZ(G_IM_FMT_CI,   G_IM_SIZ_4b)  : "ci4"   ,
            FMTSIZ(G_IM_FMT_CI,   G_IM_SIZ_8b)  : "ci8"   ,
        }
        exception_if_not(FMTSIZ(self.fmt, self.siz) in extensions, ValueError,
                         "Invalid texture format/size combination")

        return "." + extensions[FMTSIZ(self.fmt, self.siz)] + ".png"

    def to_png(self, outpath, intensity_alpha=False):
        """
        Writes this texture out to a png file. If this texture is CI and was read in from binary or from an indexed png
        it will preserve the color indices and palette of the original if it has not since been reformatted in order to
        facilitate matching in a roundtrip of bin -> png -> file -> png -> bin
        """

        # For intensity images, we don't always want to write alpha (but sometimes we do), intensity images can be
        # either RGB channel or Alpha channel but often not both. The `intensity_alpha` option determines whether to
        # write PNGs with or without alpha, both will roundtrip identically.
        has_alpha = (not self.fmt == G_IM_FMT_I) or intensity_alpha

        img = bytearray()
        if self.fmt == G_IM_FMT_CI:
            assert self.color_indices is not None , "[INTERNAL ERROR] CI textures must have color indices"
            # For matching existing CI textures, write color indices instead of color data and ensure a palette was
            # manually set
            img.extend(self.color_indices)
            # Write palette data into the PNG
            exception_if_not(self.pal is not None, Exception, "Writing CI to png must have a palette supplied")
            palette = self.pal.texels
            has_alpha = False
        else:
            # TODO are visuals exact
            texels = self.texels
            palette = None

            # write color data, possibly ignoring the alpha channel
            for px in texels:
                if not has_alpha:
                    px = px[:-1]
                img.extend(px)

        img = bytes(img)

        # write
        with open(outpath, "wb") as outfile:
            png.Writer(
                self.width, self.height, greyscale=False, alpha=has_alpha, bitdepth=8, palette=palette
            ).write_array(outfile, img)

    def to_bin(self, preswap=False):
        """
        Write to binary.
        """
        # NOTE: palette binary writing is accomplished through self.pal.to_bin()
        assert self.fmt != G_IM_FMT_CI or self.pal is not None , "[INTERNAL ERROR] CI format must have a palette"

        # Convert to target format
        enc = self.encode()

        # Pack texels
        data = bytearray()
        if self.siz == G_IM_SIZ_4b:
            data.extend([(enc[i] << 4) | enc[i + 1] for i in range(0, len(enc), 2)])
        else:
            struct_fmt = ">" + (None, "B", "H", "I")[self.siz]
            for p in enc:
                data.extend(struct.pack(struct_fmt, p))

        # TODO implement dxt=0 word swapping
        return data

    def to_c(self, preswap=False):
        """
        Write C array body (bytes)
        """
        data = self.to_bin(preswap)

        c_out = ""
        for i,b in enumerate(data,1):
            c_out += f"0x{b:02X}, "
            if i % 64 == 0:
                c_out = c_out.strip() + "\n"
        c_out = c_out.strip() + f"\n"
        return c_out

    def encode(self):
        return [self.pack(p[0] & 0xFF, p[1] & 0xFF, p[2] & 0xFF, p[3] & 0xFF) for p in self.texels]

    def unpack(self, px):
        raise NotImplementedError()

    def pack(self, r, g, b, a):
        raise NotImplementedError()



class RGBA32(Image):
    FMT = G_IM_FMT_RGBA
    SIZ = G_IM_SIZ_32b

    def unpack(self, px):
        r = (px >> 24) & 0xFF
        g = (px >> 16) & 0xFF
        b = (px >>  8) & 0xFF
        a = (px >>  0) & 0xFF

        return (r, g, b, a)

    def pack(self, r, g, b, a):
        return (r << 24) | (g << 16) | (b << 8) | a

class RGBA16(Image):
    FMT = G_IM_FMT_RGBA
    SIZ = G_IM_SIZ_16b

    def unpack(self, px):
        r = (px >> 11) & 0x1F
        r = (r << 3) | (r >> 2)

        g = (px >>  6) & 0x1F
        g = (g << 3) | (g >> 2)

        b = (px >>  1) & 0x1F
        b = (b << 3) | (b >> 2)

        a = 255 * (px & 1)

        return (r, g, b, a)

    def pack(self, r, g, b, a):
        r = r >> 3
        g = g >> 3
        b = b >> 3
        a = a != 0

        return (r << 11) | (g << 6) | (b << 1) | a

class CI4(Image):
    FMT = G_IM_FMT_CI
    SIZ = G_IM_SIZ_4b

    def encode(self):
        if self.color_indices is not None:
            return self.color_indices
        return super().encode()

    def unpack(self, px):
        return self.pal.texels[px & 0xF]

    def pack(self, r, g, b, a):
        # This only runs to convert other texture formats and pngs without palettes to CI
        exception_if_not(len(self.pal.texels) <= 16, Exception, f"palette too large for ci4: {len(self.pal.texels)}")

        pxi = pal_search((r, g, b, a), self.pal.texels)
        assert pxi != -1 , "[INTERNAL ERROR] color not in palette"
        # self.pal.texels[pxi] = self.pal.pack(r, g, b, a)
        return pxi

class CI8(Image):
    FMT = G_IM_FMT_CI
    SIZ = G_IM_SIZ_8b

    def encode(self):
        if self.color_indices is not None:
            return self.color_indices
        return super().encode()

    def unpack(self, px):
        return self.pal.texels[px & 0xFF]

    def pack(self, r, g, b, a):
        # This only runs to convert other texture formats and pngs without palettes to CI
        exception_if_not(len(self.pal.texels) <= 256, Exception, f"palette too large for ci8: {len(self.pal.texels)}")

        pxi = pal_search((r, g, b, a), self.pal.texels)
        assert pxi != -1 , "[INTERNAL ERROR] color not in palette"
        # self.pal.texels[pxi] = self.pal.pack(r, g, b, a)
        return pxi

class IA4(Image):
    FMT = G_IM_FMT_IA
    SIZ = G_IM_SIZ_4b

    def unpack(self, px):
        i = px & 0b1110
        i = (i << 4) | (i << 1) | (i >> 2)
        r = g = b = i

        a = 255 * (px & 1)

        return (r, g, b, a)

    def pack(self, r, g, b, a):
        exception_if_not(r == g and g == b, Exception, "IA4 requires greyscale rgb data")

        i = r >> 5
        a = a != 0

        return (i << 1) | a

class IA8(Image):
    FMT = G_IM_FMT_IA
    SIZ = G_IM_SIZ_8b

    def unpack(self, px):
        i = (px >> 4) & 0xF
        i = (i << 4) | i
        r = g = b = i

        a = (px >> 0) & 0xF
        a = (a << 4) | a

        return (r, g, b, a)

    def pack(self, r, g, b, a):
        exception_if_not(r == g and g == b, Exception, "IA8 requires greyscale rgb data")

        i = (r >> 4) & 0xF
        a = (a >> 4) & 0xF
        return (i << 4) | a

class IA16(Image):
    FMT = G_IM_FMT_IA
    SIZ = G_IM_SIZ_16b

    def unpack(self, px):
        i = (px >> 8) & 0xFF
        r = g = b = i

        a = (px >> 0) & 0xFF

        return (r, g, b, a)

    def pack(self, r, g, b, a):
        exception_if_not(r == g and g == b, Exception, "IA16 requires greyscale rgb data")

        i = r
        return (i << 8) | a

class I4(Image):
    FMT = G_IM_FMT_I
    SIZ = G_IM_SIZ_4b

    def unpack(self, px):
        i = px & 0xF
        i = (i << 4) | i
        r = g = b = a = i

        return (r, g, b, a)

    def pack(self, r, g, b, a):
        exception_if_not(r == g and g == b and b == a, Exception, "I4 requires greyscale image data")

        return (a >> 4) & 0xF

class I8(Image):
    FMT = G_IM_FMT_I
    SIZ = G_IM_SIZ_8b

    def unpack(self, px):
        r = g = b = a = px

        return (r, g, b, a)

    def pack(self, r, g, b, a):
        exception_if_not(r == g and g == b and b == a, Exception, "I8 requires greyscale image data")

        return a
