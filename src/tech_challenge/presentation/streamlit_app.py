from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from tech_challenge.presentation.formatting import ag_result_rows, chat_answer_text

API_URL = os.getenv("TECH_CHALLENGE_API_URL", "http://localhost:8000")


def main() -> None:
    st.set_page_config(page_title="Diagnóstico por IA", layout="wide")

    st.sidebar.title("Navegação")
    page = st.sidebar.radio(
        "Selecione a tela",
        ["Tela 1 - Diagnóstico", "Tela 2 - Explicação LLM", "Tela 3 - Resultados do AG"],
    )

    if "diagnosis" not in st.session_state:
        st.session_state.diagnosis = None
    if "llm_explanation" not in st.session_state:
        st.session_state.llm_explanation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if page == "Tela 1 - Diagnóstico":
        _diagnosis_screen()
    elif page == "Tela 2 - Explicação LLM":
        _explanation_screen()
    elif page == "Tela 3 - Resultados do AG":
        _ag_results_screen()


def _diagnosis_screen() -> None:
    st.title("Diagnóstico - Modelo Otimizado")

    try:
        metadata = _patient_metadata()
    except requests.RequestException as exc:
        st.error(f"Não foi possível carregar os pacientes da API: {exc}")
        return

    patient_index: int = (
        st.selectbox(
            "Selecionar paciente (índice do dataset)",
            range(metadata["min_index"], metadata["max_index"] + 1),
        )
        or metadata["min_index"]
    )  # type: ignore[assignment]

    if st.button("Executar Diagnóstico"):
        with st.spinner("Rodando modelo..."):
            try:
                resp = requests.post(f"{API_URL}/diagnose", json={"patient_index": patient_index}, timeout=30)
                resp.raise_for_status()
            except requests.HTTPError as exc:
                st.error(_api_error_message(exc.response))
                return
            except requests.RequestException as exc:
                st.error(f"Não foi possível conectar ao serviço: {exc}")
                return
            st.session_state.diagnosis = resp.json()
            st.session_state.llm_explanation = None
            st.session_state.chat_history = []

    if st.session_state.diagnosis:
        diagnosis: dict[str, Any] = st.session_state.diagnosis
        color = "red" if diagnosis["prediction"] == "MALIGNO" else "green"
        st.markdown(f"### Resultado: :{color}[{diagnosis['prediction']}]")
        st.progress(diagnosis["confidence"], text=f"Confiança: {diagnosis['confidence']:.0%}")

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
    st.title("Explicação Médica - Agente LLM")

    if not st.session_state.diagnosis:
        st.info("Execute um diagnóstico na Tela 1 primeiro.")
        return

    diagnosis: dict[str, Any] = st.session_state.diagnosis
    st.markdown(
        f"**Paciente selecionado** - Predição: `{diagnosis['prediction']}` | Confiança: `{diagnosis['confidence']:.0%}`"
    )

    if st.button("Gerar Explicação Médica"):
        with st.spinner("Consultando módulo de explicação..."):
            try:
                resp = requests.post(f"{API_URL}/explain", json=diagnosis, timeout=60)
                resp.raise_for_status()
            except requests.HTTPError as exc:
                st.error(_api_error_message(exc.response))
                return
            except requests.RequestException as exc:
                st.error(f"Não foi possível conectar ao serviço: {exc}")
                return
            st.session_state.llm_explanation = resp.json()

    if st.session_state.llm_explanation:
        result: dict[str, Any] = st.session_state.llm_explanation
        details = result.get("details", {})

        st.markdown("### Explicação")
        st.write(result["explanation"])

        if details.get("nivel_confianca"):
            st.markdown(f"**Nível de confiança:** {details['nivel_confianca']}")

        if details.get("recomendacoes"):
            st.markdown("### Recomendações")
            for item in details["recomendacoes"]:
                st.write(f"- {item}")

        if details.get("exames_complementares"):
            st.markdown("### Exames complementares")
            for item in details["exames_complementares"]:
                st.write(f"- {item}")

        st.warning(result["disclaimer"])

        def _copy_toast() -> None:
            st.toast("Texto copiado!")

        st.button("Copiar para relatório", on_click=_copy_toast)

        st.markdown("### Perguntas de acompanhamento")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Pergunte sobre a interpretação deste caso")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            context = {"diagnosis": diagnosis, "explanation": result}
            with st.chat_message("assistant"):
                with st.spinner("Consultando agente LLM..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/chat",
                            json={"question": question, "context": context},
                            timeout=60,
                        )
                        resp.raise_for_status()
                    except requests.HTTPError as exc:
                        answer = _api_error_message(exc.response)
                        st.error(answer)
                    except requests.RequestException as exc:
                        answer = f"Não foi possível conectar ao serviço: {exc}"
                        st.error(answer)
                    else:
                        answer = chat_answer_text(resp.json()["answer"])
                        st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


def _ag_results_screen() -> None:
    st.title("Resultados do Algoritmo Genético")

    try:
        resp = requests.get(f"{API_URL}/ag-results", timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        st.error(_api_error_message(exc.response))
        return
    except requests.RequestException as exc:
        st.error(f"Não foi possível conectar ao serviço: {exc}")
        return
    data = resp.json()

    experiments = data["experiments"]
    baseline = data["baseline"]
    best = data["best_config"]

    st.subheader("Comparativo: Baseline vs Experimentos AG")
    st.dataframe(pd.DataFrame(ag_result_rows(experiments, baseline)), width="stretch")

    st.subheader("Convergência por experimento")
    fig, ax = plt.subplots()
    for experiment in experiments:
        ax.plot(experiment["convergence"], label=experiment["name"])
    ax.axhline(baseline["metrics"]["f1"], color="gray", linestyle="--", label="Baseline F1")
    ax.set_xlabel("Geração (amostrada)")
    ax.set_ylabel("Fitness")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Melhor configuração encontrada")
    st.write(f"Experimento: {data['best_experiment']}")
    st.json(best)

    if data.get("best_model"):
        st.subheader("Resumo do melhor modelo")
        st.json(data["best_model"])


def _patient_metadata() -> dict[str, int]:
    resp = requests.get(f"{API_URL}/patients/metadata", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        "count": int(data["count"]),
        "min_index": int(data["min_index"]),
        "max_index": int(data["max_index"]),
    }


def _api_error_message(response: requests.Response | None) -> str:
    if response is None:
        return "Não foi possível conectar ao serviço."

    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text

    if response.status_code == 503:
        return f"LLM não configurado: {detail}"
    if response.status_code == 502:
        return f"Falha no provedor LLM: {detail}"
    return f"Erro da API ({response.status_code}): {detail}"


if __name__ == "__main__":
    main()
