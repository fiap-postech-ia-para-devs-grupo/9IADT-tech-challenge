from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from tech_challenge.paths import BREAST_CANCER_DATASET
from tech_challenge.presentation.formatting import (
    ag_result_rows,
    chat_answer_text,
    filter_patient_table,
    paginate_patient_table,
    patient_table_rows,
    sort_patient_table,
)

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

    _patient_selection_table(metadata)

    if st.session_state.diagnosis:
        diagnosis: dict[str, Any] = st.session_state.diagnosis
        color = "red" if diagnosis["prediction"] == "MALIGNO" else "green"
        st.markdown(f"### Resultado: :{color}[{diagnosis['prediction']}]")
        st.progress(diagnosis["confidence"], text=f"Confiança: {diagnosis['confidence']:.0%}")

        st.subheader("Principais atributos")
        features = diagnosis["top_features"]
        df = pd.DataFrame(features)
        fig, ax = plt.subplots()
        colors = ["#d62728" if impact > 0 else "#1f77b4" for impact in df["impact"]]
        ax.barh(df["feature"], df["impact"], color=colors)
        ax.set_xlabel("Impacto no modelo")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)


@st.cache_data
def _load_patient_dataset() -> pd.DataFrame:
    return pd.read_csv(BREAST_CANCER_DATASET).drop(columns=["Unnamed: 32"], errors="ignore")


def _patient_selection_table(metadata: dict[str, int]) -> None:
    st.subheader("Selecionar paciente")

    rows = patient_table_rows(_load_patient_dataset(), metadata["min_index"], metadata["max_index"])

    query = st.text_input("Buscar por ID", placeholder="Digite o ID do paciente")
    if "patient_table_sort_column" not in st.session_state:
        st.session_state.patient_table_sort_column = "Índice"
    if "patient_table_sort_ascending" not in st.session_state:
        st.session_state.patient_table_sort_ascending = True
    if "patient_table_page_size" not in st.session_state:
        st.session_state.patient_table_page_size = 25
    if "patient_table_page" not in st.session_state:
        st.session_state.patient_table_page = 1
    if st.session_state.get("patient_table_query") != query:
        st.session_state.patient_table_query = query
        st.session_state.patient_table_page = 1

    filtered_rows = filter_patient_table(rows, query)
    sorted_rows = sort_patient_table(
        filtered_rows,
        st.session_state.patient_table_sort_column,
        bool(st.session_state.patient_table_sort_ascending),
    )

    page_size = int(st.session_state.patient_table_page_size)
    total_rows = len(sorted_rows)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    st.session_state.patient_table_page = min(st.session_state.patient_table_page, total_pages)

    current_page = int(st.session_state.patient_table_page)
    page_rows = paginate_patient_table(sorted_rows, current_page, page_size)
    st.caption(f"{total_rows} pacientes encontrados")

    if page_rows.empty:
        st.info("Nenhum paciente encontrado para a busca informada.")
        return

    widths = [0.7, 1, 1, 1, 1, 1, 1, 1, 1]
    headers = page_rows.columns.tolist() + ["Ação"]
    header_cols = st.columns(widths)
    for column, header in zip(header_cols, headers):
        if header == "Ação":
            column.markdown("**Ação**")
            continue
        sort_suffix = _sort_suffix(header)
        if column.button(f"{header}{sort_suffix}", key=f"sort_patient_{header}", use_container_width=True):
            _toggle_patient_sort(header)
            st.rerun()

    for _, row in page_rows.iterrows():
        row_cols = st.columns(widths)
        for column, value in zip(row_cols[:-1], row):
            column.write(_format_patient_cell(value))

        patient_index = int(row["Índice"])
        if row_cols[-1].button("Selecionar", key=f"select_patient_{patient_index}"):
            _run_diagnosis(patient_index)

    _patient_table_footer(total_pages, total_rows)


def _sort_suffix(header: str) -> str:
    if st.session_state.patient_table_sort_column != header:
        return ""
    return " ↑" if st.session_state.patient_table_sort_ascending else " ↓"


def _toggle_patient_sort(header: str) -> None:
    if st.session_state.patient_table_sort_column == header:
        st.session_state.patient_table_sort_ascending = not st.session_state.patient_table_sort_ascending
        return
    st.session_state.patient_table_sort_column = header
    st.session_state.patient_table_sort_ascending = True
    st.session_state.patient_table_page = 1


def _patient_table_footer(total_pages: int, total_rows: int) -> None:
    st.divider()
    summary_col, page_size_col, prev_col, next_col = st.columns([2, 1.2, 1, 1])
    summary_col.caption(f"Página {st.session_state.patient_table_page} de {total_pages} · {total_rows} pacientes")
    selected_page_size = page_size_col.selectbox(
        "Linhas por página",
        [10, 25, 50, 100],
        index=[10, 25, 50, 100].index(int(st.session_state.patient_table_page_size)),
        key="patient_table_page_size_select",
    )
    if int(selected_page_size) != int(st.session_state.patient_table_page_size):
        st.session_state.patient_table_page_size = int(selected_page_size)
        st.session_state.patient_table_page = 1
        st.rerun()
    if prev_col.button("Anterior", disabled=st.session_state.patient_table_page <= 1):
        st.session_state.patient_table_page -= 1
        st.rerun()
    if next_col.button("Próxima", disabled=st.session_state.patient_table_page >= total_pages):
        st.session_state.patient_table_page += 1
        st.rerun()


def _format_patient_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _run_diagnosis(patient_index: int) -> None:
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
        st.session_state.selected_patient_index = patient_index
        st.session_state.llm_explanation = None
        st.session_state.chat_history = []


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
