"""Generate a sample VCF file for testing the parser."""

import random

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
BASES = "ACGT"

def generate_vcf(path: str, num_variants: int = 200_000) -> None:
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.3\n")
        f.write("##source=dnaparser_test_generator\n")
        f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        for i in range(num_variants):
            chrom = random.choice(CHROMS)
            pos = random.randint(1, 250_000_000)
            ref = random.choice(BASES)
            alt = random.choice([b for b in BASES if b != ref])
            info = f"DP={random.randint(5, 500)}"
            f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{random.randint(10,99)}\tPASS\t{info}\n")

    print(f"Generated {num_variants} variants -> {path}")

if __name__ == "__main__":
    generate_vcf("sample.vcf")
