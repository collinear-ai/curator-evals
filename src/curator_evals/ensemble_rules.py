
def collinear_ensemble7(row):
    if (row["output_veritas_phi"]==1 and row["output_pairwise_rm"]==1) and row["s1_tag"]==1 and row["output_veritas8b"]==1:
        return 1
    else:
        return 0

rule_map = {
    "collinear_ensemble7": collinear_ensemble7,
}

def ensemble_rule(rule_name: str): 
    return rule_map[rule_name]