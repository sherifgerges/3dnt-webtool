"""
3DNT web tool — Streamlit MVP.

Enter a gene/protein, upload two .txt files of E457K-style variants (cases and
controls), and run the 3D neighborhood test on the protein's AlphaFold structure.

Run:   SIR_REPO=/path/to/structure-informed-rvas  streamlit run app.py
"""
import os
import gzip
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import variants as V
import sir_fetch as F
import fisher_scan as FS

st.set_page_config(page_title="3DNT", layout="wide")
# base font size for the whole app (rem-based text scales from the root)
st.markdown(
    "<style>"
    "html { font-size: 100% !important; }"
    # slightly larger labels for the gene box and the two file uploaders
    ".stTextInput label p, .stFileUploader label p { font-size: 1.25rem !important; }"
    # smaller text throughout the left (sidebar) panel
    "section[data-testid='stSidebar'] label p,"
    "section[data-testid='stSidebar'] .stMarkdown,"
    "section[data-testid='stSidebar'] p,"
    "section[data-testid='stSidebar'] a { font-size: 0.85rem !important; }"
    "section[data-testid='stSidebar'] h2 { font-size: 1.0rem !important; }"
    "</style>",
    unsafe_allow_html=True,
)
_HERE = os.path.dirname(__file__)
_LOGO = os.path.join(_HERE, "logo.png")          # Finucane Lab logo
_BROAD = os.path.join(_HERE, "broad.logo.png")   # Broad Institute logo
_3DNT = os.path.join(_HERE, "3dnt_final.jpg")    # 3DNT app logo
_lc1, _lc2, _ = st.columns([2, 2, 6])
if os.path.exists(_LOGO):
    _lc1.image(_LOGO, use_container_width=True)
if os.path.exists(_BROAD):
    _lc2.image(_BROAD, use_container_width=True)
if os.path.exists(_3DNT):
    st.image(_3DNT, width=460)

# left-sidebar navigation
def _results_heading(text):
    """Slightly smaller than st.subheader, so the long results line isn't huge."""
    st.markdown(
        f"<div style='font-size:1.4rem; font-weight:600; color:#262730; "
        f"margin:8px 0 6px 0; line-height:1.3'>{text}</div>",
        unsafe_allow_html=True,
    )


def render_result(scores, meta, pdb_gz, case_v, ctrl_v, gene, acc, seq_len, radius,
                  pae_cutoff, phenotype=""):
    """Render the per-protein 3DNT result (metrics, table, stats, structure)."""
    if phenotype:
        st.markdown(
            f"<div style='font-size:1.0rem; color:#555; margin-bottom:2px'>"
            f"Phenotype: <b>{phenotype}</b></div>", unsafe_allow_html=True)
    # 5f. results ----------------------------------------------------------
    bonf = meta["bonferroni_p"]
    tested = scores[scores["center_p"].notna()].copy()
    use_fwer = meta["n_sims"] > 0 and meta["n_sig_fwer"] is not None
    if use_fwer:
        n_sig = meta["n_sig_fwer"]
        _results_heading(
            f"Results — {n_sig} significant neighborhood"
            f"{'' if n_sig == 1 else 's'} at FWER < 0.05 "
            f"(permutation, {meta['n_sims']:,} sims; {radius:g} Å radius)")
    else:
        n_sig = meta["n_sig_bonferroni"]
        _results_heading(
            f"Results — {n_sig} significant neighborhood"
            f"{'' if n_sig == 1 else 's'} at Bonferroni "
            f"(p < {bonf:.2e}; {radius:g} Å radius)")

    m1, m2, m3, m4, m5 = st.columns(5)
    if use_fwer:
        m1.metric("Pass FWER < 0.05", n_sig)
        m2.metric("Pass Bonferroni", meta["n_sig_bonferroni"])
    else:
        m1.metric("Pass Bonferroni", n_sig)
        m2.metric("Pass p < 0.01", meta["n_sig_p01"])
    m3.metric("Variants tested", meta["n_tested"])
    m4.metric("Residues analyzed", f"{meta['n_residues_kept']} / {meta['n_residues_total']}")
    m5.metric("Mean pLDDT", f"{meta['mean_plddt']:.1f}")

    # results table -------------------------------------------------------
    cols = ["aa_pos", "center_p"]
    if use_fwer:
        cols.append("fwer_p")
    cols += ["nbhd_case", "nbhd_control", "neglog10_min_p", "n_containing"]
    col_labels = {
        "aa_pos": "Residue",
        "center_p": "Neighborhood p-value",
        "fwer_p": "FWER-adjusted p (permutation)",
        "nbhd_case": "Case alleles in neighborhood",
        "nbhd_control": "Control alleles in neighborhood",
        "neglog10_min_p": "−log10(min p)",
        "n_containing": "Neighborhoods containing residue",
    }
    show = (tested.sort_values("center_p")[cols]
            .reset_index(drop=True)
            .rename(columns=col_labels))
    st.dataframe(show, use_container_width=True, height=380)
    st.download_button("Download per-residue scores TSV",
                       scores.to_csv(sep="\t", index=False),
                       file_name=(f"{gene}_{acc}"
                                  + (f"_{''.join(c if c.isalnum() else '_' for c in phenotype)}"
                                     if phenotype else "")
                                  + "_fisher_3dnt.tsv"))

    # summary stats — below the table, one point per line -----------------
    neglog_min = -np.log10(max(meta["min_p"], 1e-300)) if np.isfinite(meta["min_p"]) else float("nan")
    stat_lines = [
        f"Neighborhood radius: <b>{radius:g} Å</b>"
        + (f" · PAE cutoff {pae_cutoff:g}" if pae_cutoff > 0 else " · PAE off"),
        f"Smallest neighborhood p-value: <b>{meta['min_p']:.2e}</b> "
        f"(−log10 = {neglog_min:.2f})",
    ]
    if use_fwer:
        stat_lines += [
            f"Neighborhoods passing <b>FWER &lt; 0.05</b> "
            f"(permutation, {meta['n_sims']:,} sims): <b>{n_sig}</b>",
            f"Permutation FWER 0.05 threshold: −log10 p = "
            f"<b>{meta['neglog10_fwer_thresh']:.2f}</b> "
            f"(vs Bonferroni {meta['neglog10_bonferroni']:.2f} — "
            f"Bonferroni is conservative given correlated neighborhoods)",
            f"Neighborhoods passing Bonferroni (p &lt; {bonf:.2e}): "
            f"{meta['n_sig_bonferroni']}",
        ]
    else:
        stat_lines += [
            f"Neighborhoods passing Bonferroni (p &lt; {bonf:.2e}): <b>{n_sig}</b>",
            f"Neighborhoods passing p &lt; 0.01: <b>{meta['n_sig_p01']}</b>",
        ]
    stat_lines += [
        f"Variants tested: <b>{meta['n_tested']}</b>",
        f"Residues analyzed: <b>{meta['n_residues_kept']}</b> of "
        f"{meta['n_residues_total']} "
        f"(omitted by pLDDT filter: {meta['n_residues_omitted']}, "
        f"{meta['frac_omitted']*100:.1f}%)",
        f"Mean pLDDT: {meta['mean_plddt']:.1f} "
        f"(kept residues {meta['mean_plddt_kept']:.1f})",
        f"{meta['n_case_total']} case / {meta['n_ctrl_total']} control alleles "
        f"on kept residues",
        f"Max −log10 p: {meta['vmax']:.2f}",
    ]
    st.markdown(
        "<div style='font-size:1.05rem; line-height:1.7; color:#444'>"
        + "".join(f"• {ln}<br>" for ln in stat_lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    # 6f. structure view: continuous white → red by −log10(min p) ----------
    try:
        import py3Dmol
        with gzip.open(pdb_gz, "rt") as fh:
            pdb_text = fh.read()
        vmax = meta["vmax"] if meta["vmax"] > 0 else 1.0
        score_map = dict(zip(scores["aa_pos"].astype(int),
                             scores["neglog10_min_p"].fillna(0.0).astype(float)))

        def _white_red(t):                       # t in [0,1]: white → red
            t = max(0.0, min(1.0, t))
            g = int(round(255 * (1 - t)))
            return f"0x{255:02x}{g:02x}{g:02x}"

        view = py3Dmol.view(width="100%", height=560)
        view.addModel(pdb_text, "pdb")
        view.setStyle({"cartoon": {"color": "0xffffff"}})   # base white
        # bin residues into color buckets to limit the number of style calls
        NB = 30
        buckets = {}
        for pos, val in score_map.items():
            if val <= 0:
                continue
            b = min(NB, max(1, int(round((val / vmax) * NB))))
            buckets.setdefault(b, []).append(str(pos))
        for b, resis in buckets.items():
            view.addStyle({"resi": resis},
                          {"cartoon": {"color": _white_red(b / NB)}})
        # overlay the actual variant alleles as CA spheres (controls first so a
        # residue carrying both shows red)
        case_pos = sorted({v["pos"] for v in case_v})
        ctrl_pos = sorted({v["pos"] for v in ctrl_v})
        if ctrl_pos:
            view.addStyle({"resi": [str(p) for p in ctrl_pos], "atom": "CA"},
                          {"sphere": {"color": "0x2ca02c"}})   # green = control
        if case_pos:
            view.addStyle({"resi": [str(p) for p in case_pos], "atom": "CA"},
                          {"sphere": {"color": "0xd62728"}})   # red = case
        view.zoomTo()
        st.markdown(
            f"<div style='font-size:1.25rem; font-weight:600; margin-bottom:2px'>"
            f"{gene} · UniProt {acc} · {seq_len} aa</div>",
            unsafe_allow_html=True,
        )

        # structure on the left, compact vertical legend on its right.
        # significance tick = permutation FWER 0.05 threshold when sims ran,
        # else Bonferroni.
        if use_fwer and np.isfinite(meta["neglog10_fwer_thresh"]):
            nb, tick_label = meta["neglog10_fwer_thresh"], "— FWER 0.05"
        else:
            nb, tick_label = meta["neglog10_bonferroni"], "— Bonferroni"
        bonf_pct = (max(0.0, min(1.0, nb / vmax)) * 100
                    if (np.isfinite(nb) and vmax > 0) else None)
        v_tick = ((f"<div style='position:absolute;bottom:{bonf_pct:.1f}%;left:-3px;"
                   f"right:-3px;border-top:2px dashed #333'></div>")
                  if bonf_pct is not None else "")
        v_tick_lbl = ((f"<div style='position:absolute;bottom:{bonf_pct:.1f}%;"
                       f"transform:translateY(50%);font-size:0.8rem;color:#333'>"
                       f"{tick_label}</div>") if bonf_pct is not None else "")

        col_struct, col_leg = st.columns([4, 1])
        with col_struct:
            components.html(view._make_html(), height=580)
        with col_leg:
            st.markdown(
                "<div style='font-size:0.95rem;color:#555;margin-bottom:6px'>"
                "−log10(min&nbsp;p)<br><b>redder = more enriched</b></div>"
                "<div style='display:flex;gap:8px;height:300px;'>"
                "<div style='position:relative;width:20px;border:1px solid #ccc;"
                "background:linear-gradient(to top,#ffffff,#ff0000)'>" + v_tick + "</div>"
                "<div style='position:relative;font-size:0.85rem;color:#555;width:90px'>"
                f"<div style='position:absolute;top:0'>{vmax:.2f}</div>"
                "<div style='position:absolute;bottom:0'>0</div>"
                + v_tick_lbl +
                "</div></div>"
                "<div style='margin-top:14px;font-size:0.95rem'>"
                "<span style='color:#d62728'>&#9679;</span> Case allele<br>"
                "<span style='color:#2ca02c'>&#9679;</span> Control allele</div>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.info(f"(Structure view unavailable: {e})")



def _asd_manhattan(res):
    import altair as alt
    d = res.copy()
    d = d[d["min_p"].notna()].copy()
    if d.empty:
        st.info("No proteins in the summary."); return
    d["logp"] = d["neglog10_min_p"]
    # x order: genomic (chromosome, position) if available, else gene name
    if "order_idx" in d and "chrom_num" in d:
        d = d.sort_values("order_idx").reset_index(drop=True)
        d["x"] = d["order_idx"]
        d["cshade"] = (d["chrom_num"] % 2).astype(str)
        x_title = "Chromosome"
    else:
        d = d.sort_values("gene_name").reset_index(drop=True)
        d["x"] = range(1, len(d) + 1)
        d["cshade"] = (d["x"] % 2).astype(str)
        x_title = "Protein (ordered by gene name)"

    # threshold lines: prefer exact precomputed values (from all neighborhoods)
    lines = []
    if "fdr_line" in d and np.isfinite(d["fdr_line"].iloc[0]):
        lines.append(("FDR < 0.05", float(d["fdr_line"].iloc[0]), [6, 4]))
    elif "fdr" in d and (d["fdr"] < 0.05).any():
        lines.append(("FDR < 0.05", float(-np.log10(d.loc[d["fdr"] < 0.05, "min_p"].max())), [6, 4]))
    if "fwer_line" in d and np.isfinite(d["fwer_line"].iloc[0]):
        lines.append(("FWER < 0.05", float(d["fwer_line"].iloc[0]), [2, 3]))
    elif "fwer" in d and d["fwer"].notna().any() and (d["fwer"] < 0.05).any():
        lines.append(("FWER < 0.05", float(-np.log10(d.loc[d["fwer"] < 0.05, "min_p"].max())), [2, 3]))

    ymax = float(d["logp"].max())
    pts = alt.Chart(d).mark_circle(size=50, opacity=0.85).encode(
        x=alt.X("x:Q", title=x_title, axis=alt.Axis(labels=False, ticks=False),
                scale=alt.Scale(domain=[0, len(d) + 1])),
        y=alt.Y("logp:Q", title="−log₁₀(p)", scale=alt.Scale(domain=[-0.6, ymax * 1.08])),
        color=alt.Color("cshade:N", scale=alt.Scale(domain=["0", "1"],
                        range=["#2c7fb8", "#7fcdbb"]), legend=None),
        tooltip=["gene_name", "uniprot_id", "chrom", "top_aa_pos",
                 alt.Tooltip("min_p:Q", format=".2e"),
                 alt.Tooltip("logp:Q", title="-log10 p", format=".2f"), "n_tested"])
    layers = [pts]
    # threshold lines as a single layer -> gives a linetype legend on top,
    # dashed = FDR, dotted = FWER, grey30 (like the ggplot)
    if lines:
        ldf = pd.DataFrame([{"level": lab, "y": y} for lab, y, _ in lines])
        layers.append(alt.Chart(ldf).mark_rule(color="#4d4d4d").encode(
            y="y:Q",
            strokeDash=alt.StrokeDash("level:N",
                scale=alt.Scale(domain=["FDR < 0.05", "FWER < 0.05"],
                                range=[[6, 4], [2, 3]]),
                legend=alt.Legend(orient="top", title=None, symbolType="stroke"))))
    # chromosome tick labels along the bottom
    if "chrom_num" in d:
        cm = (d.groupby("chrom_num").agg(x=("x", "mean"), chrom=("chrom", "first"))
              .reset_index())
        cm["lab"] = cm["chrom"].str.replace("chr", "", regex=False)
        cm["y"] = -0.5
        layers.append(alt.Chart(cm).mark_text(fontSize=9, color="#555").encode(
            x="x:Q", y="y:Q", text="lab"))
    # label the FWER-significant genes (ggrepel-style: offset up, thin connector)
    sig = d[d["fwer"] < 0.05] if ("fwer" in d and d["fwer"].notna().any()) else d[d["fdr"] < 0.05]
    if len(sig):
        lab = sig.copy()
        lab["ytext"] = lab["logp"] + 0.55
        layers.append(alt.Chart(lab).mark_rule(color="#999", size=0.6).encode(
            x="x:Q", y="logp:Q", y2="ytext:Q"))
        layers.append(alt.Chart(lab).mark_text(fontSize=11, color="black", dy=-4).encode(
            x="x:Q", y="ytext:Q", text="gene_name"))
    chart = (alt.layer(*layers).properties(height=460)
             .configure_axis(grid=False, domainColor="#333", labelColor="#333",
                             titleColor="#333", titleFontSize=13)
             .configure_view(strokeWidth=0)
             .configure_legend(labelFontSize=12))
    st.altair_chart(chart, use_container_width=True)
    if lines:
        st.caption(" · ".join(f"{lab_}: −log10 p = {y:.2f}" for lab_, y, _ in lines)
                   + f" · {len(sig)} gene(s) labeled")
    st.markdown(
        "<div style='text-align:center; font-size:0.9rem; color:#555; margin-top:2px'>"
        "Figure 2 from Nason <i>et al.</i>, medRxiv 2026.</div>",
        unsafe_allow_html=True,
    )


def render_asd_page(radius, pae_cutoff, plddt_cutoff, n_sims):
    st.title("ASD — genome-wide 3D neighborhood test")
    asd_dir = os.path.join(_HERE, "asd_data")
    var_path = os.path.join(asd_dir, "ASD_variants.tsv")
    # Manhattan uses the paper's published results (exact); fall back to a locally
    # recomputed summary if the paper file isn't bundled.
    paper_path = os.path.join(asd_dir, "asd_paper_manhattan.tsv")
    res_path = paper_path if os.path.exists(paper_path) else os.path.join(asd_dir, "asd_web_results.tsv")
    if not os.path.exists(var_path):
        st.error("Bundled ASD variants not found (asd_data/ASD_variants.tsv)."); return
    variants = pd.read_csv(var_path, sep="\t")
    st.markdown(
        "Manhattan = the **3DNT paper's published ASD results** (each point is a "
        "protein's most significant neighborhood; dashed = FDR 0.05, dotted = "
        "FWER 0.05). Hover for gene. Selecting a protein below **recomputes it live** "
        "with this site's Fisher engine — top genes match the paper, but values can "
        "differ slightly because the site fetches AlphaFold2 models rather than the "
        "paper's AlphaFold3 structures.")
    res = None
    if os.path.exists(res_path):
        res = pd.read_csv(res_path, sep="\t")
        _asd_manhattan(res)
    else:
        st.info("Manhattan summary not found. Expected asd_data/asd_paper_manhattan.tsv. "
                "You can still open individual proteins below.")
    st.markdown("---")
    st.subheader("Individual protein")
    genes = sorted(variants["gene_name"].dropna().unique().tolist())
    default_ix = 0
    if res is not None and len(res):
        tg = res.sort_values("min_p").iloc[0]["gene_name"]
        if tg in genes:
            default_ix = genes.index(tg)
    gene = st.selectbox(
        f"Select a gene to recompute live ({len(genes)} proteins in the ASD analysis)",
        genes, index=default_ix)
    if not st.button(f"Run 3DNT for {gene}", type="primary"):
        return
    sub = variants[variants["gene_name"] == gene]
    acc = str(sub["uniprot_id"].iloc[0])
    counts = sub.groupby("aa_pos")[["ac_case", "ac_control"]].sum().reset_index()
    counts["uniprot_id"] = acc
    try:
        matches = F.resolve_uniprot(acc)
        seq = matches[0]["sequence"] if matches else ""
    except Exception:
        seq = ""
    try:
        pdb_gz, pae = F.fetch_alphafold(acc, seq_len=len(seq) or None)
    except Exception as e:
        st.error(f"AlphaFold fetch failed for {acc}: {e}"); return
    spin = (f"Running Fisher 3DNT for {gene} + {n_sims:,} permutations…"
            if n_sims > 0 else f"Running Fisher 3DNT for {gene}…")
    # pLDDT filter off here to match the paper's ASD analysis (it keeps disordered
    # residues; filtering them strips control alleles from the background)
    with st.spinner(spin):
        try:
            scores, meta = FS.run_fisher_3dnt(
                counts, pdb_gz, pae, radius=radius, pae_cutoff=pae_cutoff,
                plddt_cutoff=0.0, n_sims=int(n_sims))
        except Exception as e:
            st.error(f"Scan failed: {e}"); return
    case_v = [{"pos": int(p)} for p in counts.loc[counts.ac_case > 0, "aa_pos"]]
    ctrl_v = [{"pos": int(p)} for p in counts.loc[counts.ac_control > 0, "aa_pos"]]
    render_result(scores, meta, pdb_gz, case_v, ctrl_v, gene, acc,
                  len(seq) or int(counts["aa_pos"].max()), radius, pae_cutoff,
                  phenotype="Autism (ASD)")


st.markdown(
    "<div style='font-size:1.15rem; font-weight:600; color:#333; margin-bottom:4px'>"
    "What would you like to do?</div>",
    unsafe_allow_html=True,
)
_MODES = {
    "Upload my own variants and gene": "3DNT",
    "Reproduce data from the paper (Autism)": "ASD (genome-wide)",
    "Citations": "Citations",
}
_choice = st.radio("mode", list(_MODES), horizontal=True, label_visibility="collapsed")
page = _MODES[_choice]
st.markdown("<hr style='margin:6px 0 14px 0; border:none; border-top:1px solid #eee'>",
            unsafe_allow_html=True)

if page == "Citations":
    st.title("Citations")
    st.markdown("If you use this tool in your work, please cite:")

    st.markdown(
        "**Genetic and structural evidence links Ca²⁺ dysregulation and ATP2B2 to "
        "neuropsychiatric illness**  \n"
        "Sherif Gerges, Nikolaj Catois Straarup, Mohamed A. El-Brolosy, F. Kyle "
        "Satterstrom, Nolan Kamitaki, Jiayi Yuan, Emi Ling, Carmen Gelze, Raozhou "
        "Lin, Melissa Goldman, Curtis Mello, Tarjinder Singh, The Autism Sequencing "
        "Consortium, Jonathan S. Weissman, Sabina Berretta, Jen Q. Pan, Hilary "
        "Finucane, Charlott Stock, Poul Nissen, Steven A. McCarroll, Mark Daly.  \n"
        "*bioRxiv* 2025.08.25.672202; doi: "
        "[https://doi.org/10.1101/2025.08.25.672202]"
        "(https://doi.org/10.1101/2025.08.25.672202)"
    )

    st.markdown("---")

    st.markdown(
        "**Systematic identification of disease-associated 3D neighborhoods in "
        "protein structures**  \n"
        "Emily Nason, Sherif Gerges, F. Kyle Satterstrom, Bram L. Gorissen, Ruqi "
        "Liao, Georgia Panagiotaropoulou, Jeremy Guez, The Autism Sequencing "
        "Consortium, Konrad J. Karczewski, Mark Daly, Hilary Finucane.  \n"
        "*medRxiv* 2026.05.29.26354366; doi: "
        "[https://doi.org/10.64898/2026.05.29.26354366]"
        "(https://doi.org/10.64898/2026.05.29.26354366)"
    )
    st.stop()

if page == "3DNT":
    st.title("3D Neighborhood Test")
    st.markdown(
        "<div style='font-size:1.02rem; line-height:1.5; color:#444; max-width:900px; "
        "margin-bottom:10px'>"
        "The 3D neighborhood test maps <b>case</b> and <b>control</b> missense "
        "variants onto a protein's AlphaFold structure and asks whether case "
        "variants cluster together in 3D. For every residue it defines a spatial "
        "<b>neighborhood</b> — all residues within a radius <i>R</i> (using all-atom "
        "distances, down-weighting poorly-predicted pairs via AlphaFold PAE) — and "
        "compares the case:control ratio inside that neighborhood to the rest of the "
        "protein with a one-sided <b>Fisher's exact test</b>. Each residue is scored "
        "by the strongest enrichment of any neighborhood containing it "
        "(−log₁₀ of the smallest p), and significance is assessed by <b>permutation</b> "
        "(shuffling case/control labels) to account for the many overlapping, "
        "correlated neighborhoods. Enter a gene and upload case + control variant "
        "lists below.</div>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("Parameters")
    radius = st.slider("Neighborhood radius (Å)", 6.0, 30.0, 15.0, 1.0)
    pae_cutoff = st.number_input("PAE cutoff (0 = off)", 0.0, 35.0, 15.0, 1.0)
    st.markdown(
        "<a href='https://www.ebi.ac.uk/training/online/courses/alphafold/"
        "inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-"
        "confidence-scores/pae-a-measure-of-global-confidence-in-alphafold-"
        "predictions/' target='_blank' style='font-size:0.95rem'>What is PAE?</a>",
        unsafe_allow_html=True,
    )
    plddt_cutoff = st.slider("pLDDT cutoff", 0.0, 90.0, 50.0, 5.0)
    st.markdown(
        "<a href='https://www.ebi.ac.uk/training/online/courses/alphafold/"
        "inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-"
        "confidence-scores/plddt-understanding-local-confidence/' "
        "target='_blank' style='font-size:0.95rem'>What is pLDDT?</a>",
        unsafe_allow_html=True,
    )
    n_sims = st.select_slider(
        "Null simulations (multiple-testing correction)",
        options=[0, 100, 500, 1000, 5000, 10000], value=1000)
    st.markdown("---")
    if n_sims > 0:
        st.caption("Per-neighborhood **one-sided Fisher's exact test** (case "
                   "enrichment). Significance uses a **permutation FWER** "
                   "(Westfall–Young min-P over the correlated neighborhoods): "
                   "case/control labels are shuffled across alleles, and a "
                   "center is called significant when its p beats the null of "
                   "the best center in ≥95% of simulations. Residues are colored "
                   "by −log10 of the smallest p among neighborhoods containing "
                   "them. Within-protein, not genome-wide.")
    else:
        st.caption("Per-neighborhood **one-sided Fisher's exact test** (case "
                   "enrichment). With simulations off, only the **Bonferroni** "
                   "cutoff (0.05 / #tested) is shown — conservative here because "
                   "the neighborhoods are correlated. Residues are colored by "
                   "−log10 of the smallest p among neighborhoods containing them. "
                   "Within-protein, not genome-wide.")

if page == "ASD (genome-wide)":
    render_asd_page(radius, pae_cutoff, plddt_cutoff, n_sims)
    st.stop()

query = st.text_input("Gene symbol or UniProt accession", placeholder="e.g. CDK13 or Q14004")
_PHENOS = ["Autism (ASD)", "Schizophrenia (SCZ)", "Developmental disorder",
           "Epilepsy", "Intellectual disability", "Other / custom"]
phenotype = st.selectbox(
    "Phenotype", _PHENOS, index=0,
    help="Label for this analysis (shown in the results; does not change the computation).")
if phenotype == "Other / custom":
    phenotype = st.text_input("Enter phenotype label", value="").strip() or "Custom"
c1, c2 = st.columns(2)
case_file = c1.file_uploader("Case variants", type=["txt", "tsv", "csv"])
ctrl_file = c2.file_uploader("Control variants", type=["txt", "tsv", "csv"])
st.markdown(
    "<div style='font-size:1.05rem; line-height:1.4; color:#555'>"
    "<code>.txt</code>, <code>.tsv</code>, or <code>.csv</code>. One variant per line, "
    "either notation (auto-detected): <code>A985V</code> or <code>p.Ala985Val</code>. "
    "In a .tsv/.csv the variant can be in any column; an integer column is used as an "
    "allele count (otherwise each line = one allele). Header rows are ignored.</div>",
    unsafe_allow_html=True,
)

# --- submit-your-own-structure (placeholder) --------------------------------
if st.button("I want to submit my own structure"):
    st.info("Uploading your own structure (a PDB file) is coming soon. For now, the "
            "app automatically fetches and uses the AlphaFold model for the "
            "gene / UniProt accession you enter above.")

# --- bundled example dataset ------------------------------------------------
EX_DIR = os.path.join(_HERE, "examples")
EX_CASE = os.path.join(EX_DIR, "ATP2B2_cases.txt")
EX_CTRL = os.path.join(EX_DIR, "ATP2B2_controls.txt")
ex_available = os.path.exists(EX_CASE) and os.path.exists(EX_CTRL)

ex_run = False
if ex_available:
    ex_run = st.button(
        ":red[**Try ATP2B2 example (using autism and schizophrenia missense variants)**]",
        help="Load the bundled ATP2B2 (Q01814) case/control set and run")

# resolve inputs: example data takes precedence when its button is clicked
if ex_run:
    query_val = "ATP2B2"
    case_bytes = open(EX_CASE, "rb").read()
    ctrl_bytes = open(EX_CTRL, "rb").read()
else:
    query_val = query
    case_bytes = case_file.getvalue() if case_file else None
    ctrl_bytes = ctrl_file.getvalue() if ctrl_file else None

run_clicked = st.button("Run 3DNT", type="primary",
                        disabled=not (query and case_file and ctrl_file))
run = run_clicked or (ex_run and case_bytes is not None and ctrl_bytes is not None)

if run:
    # 1. resolve protein -------------------------------------------------------
    try:
        matches = F.resolve_uniprot(query_val)
    except Exception as e:
        st.error(f"UniProt lookup failed: {e}"); st.stop()
    if not matches:
        st.error("No human reviewed UniProt entry found for that query."); st.stop()
    if len(matches) > 1:
        label = st.selectbox("Multiple matches — pick one:",
                             [f"{m['gene']} ({m['accession']}, {len(m['sequence'])} aa)" for m in matches])
        chosen = matches[[f"{m['gene']} ({m['accession']}, {len(m['sequence'])} aa)" for m in matches].index(label)]
    else:
        chosen = matches[0]
    acc, seq, gene = chosen["accession"], chosen["sequence"], chosen["gene"]
    st.success(f"{gene}  ·  {acc}  ·  {len(seq)} aa")

    # 2. parse variants --------------------------------------------------------
    case_v, case_err, case_fmt = V.parse_variants(case_bytes.decode("utf-8", "ignore"))
    ctrl_v, ctrl_err, ctrl_fmt = V.parse_variants(ctrl_bytes.decode("utf-8", "ignore"))

    def fmt_msg(fmt):
        if not fmt:
            return "no variants"
        return " + ".join(f"{n} {k}" for k, n in fmt.items())
    st.info(f"Detected notation — cases: {fmt_msg(case_fmt)}; controls: {fmt_msg(ctrl_fmt)}. "
            "Both `A985V` and `p.Ala985Val` are accepted.")

    for name, err in (("case", case_err), ("control", ctrl_err)):
        if err:
            st.warning(f"{len(err)} unparseable {name} line(s) skipped, e.g. line {err[0][0]}: '{err[0][1]}'")
    warns = V.validate_against_sequence(case_v, seq) + V.validate_against_sequence(ctrl_v, seq)
    if warns:
        with st.expander(f"⚠️ {len(warns)} ref-AA / position mismatches (click to review)"):
            for w in warns[:200]:
                st.text(w)
    df_counts = V.build_counts(case_v, ctrl_v, acc)
    n_case = sum(v.get("count", 1) for v in case_v)
    n_ctrl = sum(v.get("count", 1) for v in ctrl_v)
    st.write(f"{n_case} case alleles, {n_ctrl} control alleles "
             f"across {len(df_counts)} residues.")

    # 3. fetch structure -------------------------------------------------------
    try:
        pdb_gz, pae = F.fetch_alphafold(acc, seq_len=len(seq))
    except Exception as e:
        st.error(f"AlphaFold fetch failed: {e}"); st.stop()

    # 4f. run the Fisher neighborhood test ---------------------------------
    _spin = ("Running Fisher 3D neighborhood test + "
             f"{n_sims:,} permutations…") if n_sims > 0 else \
            "Running Fisher 3D neighborhood test…"
    with st.spinner(_spin):
        try:
            scores, meta = FS.run_fisher_3dnt(
                df_counts, pdb_gz, pae, radius=radius,
                pae_cutoff=pae_cutoff, plddt_cutoff=plddt_cutoff,
                n_sims=int(n_sims))
        except Exception as e:
            st.error(f"Fisher scan failed: {e}"); st.stop()

    render_result(scores, meta, pdb_gz, case_v, ctrl_v, gene, acc,
                  len(seq), radius, pae_cutoff, phenotype=phenotype)
