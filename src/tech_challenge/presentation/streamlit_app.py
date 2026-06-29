from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"


def main() -> None:
    st.set_page_config(page_title="Diagnostico por IA", layout="wide")

    st.sidebar.title("Navegacao")
    page = st.sidebar.radio(
        "Selecione a tela",
        ["Tela 1 - Diagnostico", "Tela 2 - Explicacao LLM", "Tela 3 - Resultados do AG"],
    )

    if "diagnosis" not in st.session_state:
        st.session_state.diagnosis = None

    if page == "Tela 1 - Diagnostico":
        _diagnosis_screen()
    elif page == "Tela 2 - Explicacao LLM":
        _explanation_screen()
    elif page == "Tela 3 - Resultados do AG":
        _ag_results_screen()


def _diagnosis_screen() -> None:
    st.title("Diagnostico - Modelo Otimizado")

    patient_index: int = st.selectbox("Selecionar paciente (indice do dataset)", range(0, 569)) or 0  # type: ignore[assignment]

    if st.button("Executar Diagnostico"):
        with st.spinner("Rodando modelo..."):
            resp = requests.post(f"{API_URL}/diagnose", json={"patient_index": patient_index}, timeout=30)
            resp.raise_for_status()
            st.session_state.diagnosis = resp.json()

    if st.session_state.diagnosis:
        diagnosis: dict[str, Any] = st.session_state.diagnosis
        color = "red" if diagnosis["prediction"] == "MALIGNO" else "green"
        st.markdown(f"### Resultado: :{color}[{diagnosis['prediction']}]")
        st.progress(diagnosis["confidence"], text=f"Confianca: {diagnosis['confidence']:.0%}")

        st.subheader("Top features")
        features = diagnosis["top_features"]
        df = pd.DataFrame(features)
        fig, ax = plt.subplots()
        colors = ["#d62728" if impact > 0 else "#1f77b4" for impact in df["impact"]]
        ax.barh(df["feature"], df["impact"], color=colors)
        ax.set_xlabel("Impacto no modelo")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)


def _explanation_screen() -> None:
    st.title("Explicacao Medica - Agente LLM")

    if not st.session_state.diagnosis:
        st.info("Execute um diagnostico na Tela 1 primeiro.")
        return

    diagnosis: dict[str, Any] = st.session_state.diagnosis
    st.markdown(
        f"**Paciente selecionado** - Predicao: `{diagnosis['prediction']}` | "
        f"Confianca: `{diagnosis['confidence']:.0%}`"
    )

    if st.button("Gerar Explicacao Medica"):
        with st.spinner("Consultando modulo de explicacao..."):
            resp = requests.post(f"{API_URL}/explain", json=diagnosis, timeout=60)
            resp.raise_for_status()
            result = resp.json()

        st.markdown("### Explicacao")
        st.write(result["explanation"])
        st.warning(result["disclaimer"])

        def _copy_toast() -> None:
            st.toast("Texto copiado!")

        st.button("Copiar para relatorio", on_click=_copy_toast)


def _ag_results_screen() -> None:
    st.title("Resultados do Algoritmo Genetico")

    resp = requests.get(f"{API_URL}/ag-results", timeout=30)
    resp.raise_for_status()
    data = resp.json()

    experiments = data["experiments"]
    baseline = data["baseline_f1"]
    best = data["best_config"]

    st.subheader("Comparativo: Baseline vs Experimentos AG")
    rows = [{"Modelo": "RF Baseline", "F1": baseline, "Pop": "-", "Gen": "-", "Mut": "-"}]
    for experiment in experiments:
        rows.append(
            {
                "Modelo": experiment["name"],
                "F1": experiment["best_f1"],
                "Pop": experiment["population"],
                "Gen": experiment["generations"],
                "Mut": f"{experiment['mutation_rate']:.0%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Convergencia por experimento")
    fig, ax = plt.subplots()
    for experiment in experiments:
        ax.plot(experiment["convergence"], label=experiment["name"])
    ax.axhline(baseline, color="gray", linestyle="--", label="Baseline")
    ax.set_xlabel("Geracao (amostrada)")
    ax.set_ylabel("F1-score")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Melhor configuracao encontrada")
    st.json(best)


if __name__ == "__main__":
    main()
