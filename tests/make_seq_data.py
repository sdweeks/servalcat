"""
Synthetic polymer models (protein, DNA, RNA) with unmodelled residues, for testing sequence assignment.
Backbone-only atoms with proper bond lengths between neighbours, so that residues are connected
unless numbering skips a residue (a real chain break).
"""
import random
import servalcat # must be imported before gemmi (registers the bundled gemmi)
import gemmi

AA = "ACDEFGHIKLMNPQRSTVWY"
KINDS = {"AA": gemmi.ResidueKind.AA, "DNA": gemmi.ResidueKind.DNA, "RNA": gemmi.ResidueKind.RNA}

def random_protein(n, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice(AA) for _ in range(n))

def chain(name, seq, kind, missing=(), names=None, shift=None):
    """seq: construct sequence; missing: 1-based positions that are not modelled; names: {position: residue name}
    overriding the name; shift: (first, last, d): residues first..last are those of seq[pos+d-1] (register error)"""
    return dict(name=name, seq=seq, kind=kind, missing=set(missing), names=names or {}, shift=shift)

def build(chains):
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(80, 80, 80, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model(1)
    for k, c in enumerate(chains):
        ch = gemmi.Chain(c["name"])
        for i, aa in enumerate(c["seq"], start=1):
            if i in c["missing"]:
                continue
            if c["shift"] and c["shift"][0] <= i <= c["shift"][1]:
                aa = c["seq"][i + c["shift"][2] - 1]
            r = gemmi.Residue()
            r.name = c["names"].get(i) or gemmi.expand_one_letter(aa, KINDS[c["kind"]])
            r.seqid = gemmi.SeqId(i, " ")
            r.entity_type = gemmi.EntityType.Polymer
            if c["kind"] == "AA":   # C(i)-N(i+1) = 1.4 A
                atoms = [("N", 0.0), ("CA", 1.46), ("C", 2.4), ("O", 2.4)]; step = 3.8
            else:                   # O3'(i)-P(i+1) = 1.6 A
                atoms = [("P", 0.0), ("O5'", 1.6), ("C5'", 2.3), ("C4'", 3.1), ("C3'", 3.7), ("O3'", 4.4)]; step = 6.0
            for an, dx in atoms:
                a = gemmi.Atom()
                a.name = an
                a.element = gemmi.Element("C" if an == "CA" else an[0])
                a.pos = gemmi.Position(step * i + dx, 10 * k, 1.2 if an == "O" else 0.0)
                a.occ = 1.0
                a.b_iso = 20.0
                r.add_atom(a)
            ch.add_residue(r)
        model.add_chain(ch)
    st.add_model(model)
    return st

def four_chains():
    """returns (chains, fasta_records): two proteins, DNA and RNA (same base order) with gaps"""
    P1 = random_protein(30, 1)[:4] + "M" + random_protein(30, 1)[5:]   # Met at 5
    P2 = random_protein(26, 2)
    DNA, RNA = "GCGATTAGCGCT", "GCGAUUAGCGCU"
    chains = [chain("A", P1, "AA", [1, 2, 11, 12, 13, 14]), chain("B", P2, "AA", [9, 10, 11, 25, 26]),
              chain("C", DNA, "DNA", [11, 12]), chain("D", RNA, "RNA", [6, 7])]
    return chains, [("P1", P1), ("P2", P2), ("DNA", DNA), ("RNA", RNA),
                    ("decoy_prot", random_protein(28, 3)), ("decoy_dna", "TTTTAAAACCCC")]

def write_fasta(path, records, description="some description"):
    with open(path, "w") as f:
        for n, s in records:
            f.write(">{} {}\n{}\n".format(n, description, s))
