# ============================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, timedelta
from scipy.stats import jarque_bera, linregress
from scipy.optimize import minimize
import plotly.colors as pc
import plotly.graph_objects as go
import io
import tempfile
import os
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Herramienta de Visualización Financiera | Finanzas I",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función Auxiliar para Tablas de Alta Resolución en PDF
def create_pdf_table(df, header_color='#002060', cell_color='#F8F9FA'):
    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{col}</b>" for col in df.columns], fill_color=header_color, font=dict(color='white', size=13, family="Arial"), align='center', height=32),
        cells=dict(values=[df[col] for col in df.columns], fill_color=cell_color, font=dict(color='black', size=11, family="Arial"), align='center', height=27)
    )])
    calc_height = len(df) * 27 + 32 + 10
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=calc_height)
    return fig

# ============================================
# TÍTULO PRINCIPAL
# ============================================
st.title("Herramienta de Visualización Financiera")
st.markdown("**Finanzas I - UPB La Paz | Gestión 2026**")
st.divider()

# ============================================
# BARRA LATERAL (CONFIGURACIÓN GLOBAL)
# ============================================
st.sidebar.header("Configuración General")

usar_ticker_rf = st.sidebar.checkbox("Seleccionar Ticker de bono (Tasa Libre de Riesgo)", value=False, key="chk_rf_global")

if usar_ticker_rf:
    rf_ticker = st.sidebar.selectbox(
        "Activo Libre de Riesgo (Risk Free)",
        ["^IRX (T-Bill 13 semanas)", "^TNX (Bono Tesoro 10 años)", "^FVX (Bono Tesoro 5 años)", "IEF (ETF Bonos 7-10 años)"],
        index=1, key="sel_rf_ticker" 
    )
    rf_rate_manual = 0.05 
else:
    rf_rate_manual = st.sidebar.number_input(
        "Tasa Libre de Riesgo anual (%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=5.0, 
        step=0.5,
        key="num_rf_manual",
        help="Tasa libre de riesgo utilizada para calcular ratios ajustados por riesgo"
    ) / 100
    rf_ticker = None

benchmark_ticker = st.sidebar.selectbox(
    "Índice de Referencia (Benchmark)",
    ["^GSPC", "^IXIC", "SPY", "EWZ"],
    index=0,
    key="sel_bench_global",
    help="Utilizado como proxy de mercado"
)

freq_opciones = {"Diaria (252)": 252, "Semanal (52)": 52, "Mensual (12)": 12, "Anual (1)": 1}
frecuencia_sel = st.sidebar.selectbox(
    "Frecuencia de los datos", 
    list(freq_opciones.keys()), 
    index=0, 
    key="sel_freq",
    help="Define el factor de anualización para los cálculos de retorno y volatilidad."
)
ann_factor = freq_opciones[frecuencia_sel]

intervalo_yf = {"Diaria (252)": "1d", "Semanal (52)": "1wk", "Mensual (12)": "1mo", "Anual (1)": "1y"}
intervalo_seleccionado = intervalo_yf[frecuencia_sel]

# ============================================
# MOTOR DE DESCARGA GLOBAL Y FUNCIONES
# ============================================
if usar_ticker_rf:
    ticker_clean = rf_ticker.split(" ")[0]
    if st.session_state.get('rf_ticker_name') != ticker_clean or st.session_state.get('rf_ticker_data') is None:
        with st.sidebar.status(f"Descargando {ticker_clean}...", expanded=False):
            try:
                rf_d = yf.download(ticker_clean, period="max", progress=False)['Close']
                if isinstance(rf_d, pd.DataFrame):
                    rf_d = rf_d.iloc[:, 0]
                st.session_state['rf_ticker_data'] = rf_d
                st.session_state['rf_ticker_name'] = ticker_clean
            except:
                st.session_state['rf_ticker_data'] = None

def get_dynamic_rf(prices_df):
    if usar_ticker_rf and st.session_state.get('rf_ticker_data') is not None:
        try:
            rf_series = st.session_state['rf_ticker_data'].loc[prices_df.index.min():prices_df.index.max()]
            return rf_series.mean() / 100
        except:
            return rf_rate_manual
    return rf_rate_manual

# ============================================
# PESTAÑAS
# ============================================
tab1, tab2, tab3 = st.tabs([
    "Módulo 1: Análisis de Riesgo y Rentabilidad", 
    "Módulo 2: Optimización de Portafolios", 
    "Módulo 3: CAPM y SML"
])

# ============================================
# MÓDULO 1: ANÁLISIS DE RIESGO Y RENTABILIDAD
# ============================================
with tab1:
    st.header("Módulo 1: Análisis de Riesgo y Rentabilidad")

    st.subheader("1. Carga de Datos")

    data_source = st.radio(
        "Fuente de datos",
        ["Descargar de Yahoo Finance", "Subir archivo CSV/Excel (Refinitiv)"],
        horizontal=True,
        key="rad_source_m1"
    )

    if 'prices_full' not in st.session_state:
        st.session_state['prices_full'] = None
        st.session_state['volume_full'] = None
        st.session_state['benchmark_data'] = None
        st.session_state['current_benchmark'] = None

    if data_source == "Descargar de Yahoo Finance":
        st.info("Ingresar los tickers separados por coma. Ejemplo: AAPL,MSFT,GOOGL,AMZN,TSLA,JPM")

        tickers_text = st.text_input(
            "Tickers (separados por coma)",
            value="",
            placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA, JPM",
            key="txt_tickers_m1"
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Fecha de inicio", value=date(2020, 1, 1), key="date_start_m1")
        with col2:
            end_date = st.date_input("Fecha de fin", value=date.today(), key="date_end_m1")
        
        use_max_period = st.checkbox("Usar el período máximo histórico disponible", value=False, key="chk_maxper_m1")

        if st.button("Descargar datos", type="primary", key="btn_download_m1"):
            tickers_list = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
            
            if len(tickers_list) == 0:
                st.warning("Se requiere ingresar al menos un ticker.")
            else:
                with st.spinner("Descargando matriz de precios y volumen..."):
                    try:
                        end_date_yf = end_date + timedelta(days=1)
                        if use_max_period:
                            raw_data = yf.download(tickers_list, period="max", interval=intervalo_seleccionado, auto_adjust=True, progress=False)
                        else:
                            raw_data = yf.download(tickers_list, start=start_date, end=end_date_yf, interval=intervalo_seleccionado, auto_adjust=True, progress=False)
                        
                        if isinstance(raw_data.columns, pd.MultiIndex):
                            prices_data = raw_data['Close']
                            volume_data = raw_data['Volume'] if 'Volume' in raw_data else None
                        else:
                            prices_data = raw_data[['Close']].copy()
                            prices_data.columns = tickers_list
                            volume_data = raw_data[['Volume']].copy() if 'Volume' in raw_data else None
                            if volume_data is not None:
                                volume_data.columns = tickers_list
                        
                        st.session_state['last_uploaded_file'] = None
                        st.session_state['prices_full'] = prices_data
                        st.session_state['volume_full'] = volume_data
                        st.session_state['current_benchmark'] = benchmark_ticker
                        
                        try:
                            if use_max_period:
                                bench_raw = yf.download(benchmark_ticker, period="max", auto_adjust=True, progress=False)
                            else:
                                bench_raw = yf.download(benchmark_ticker, start=start_date, end=end_date_yf, auto_adjust=True, progress=False)
                            
                            if isinstance(bench_raw.columns, pd.MultiIndex):
                                bench = bench_raw['Close'].copy()
                            elif 'Close' in bench_raw.columns:
                                bench = bench_raw[['Close']].copy()
                            else:
                                bench = bench_raw.copy()
                            
                            if isinstance(bench, pd.Series):
                                bench = bench.to_frame()
                            bench.columns = [benchmark_ticker]
                            
                            st.session_state['benchmark_data'] = bench
                        except:
                            st.session_state['benchmark_data'] = None
                        
                        st.success(f"Procesamiento exitoso: {len(tickers_list)} activos descargados.")
                    except Exception as e:
                        st.error(f"Falla en la descarga: {e}")

    else:
        st.info("Formato requerido: La primera columna debe estructurar las fechas del análisis. A partir de la segunda columna se deben presentar los precios de cierre. La primera fila establecerá los identificadores (Tickers).")
        uploaded_file = st.file_uploader("Subir base de datos (CSV o Excel)", type=["csv", "xlsx", "xls"], key="up_file_m1")

        if uploaded_file is not None:
            file_hash = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if st.session_state.get('last_uploaded_file') != file_hash:
                with st.spinner("Compilando estructura de la base de datos..."):
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            data = pd.read_csv(uploaded_file, index_col=0)
                        else:
                            data = pd.read_excel(uploaded_file, index_col=0)
                        
                        data.index = pd.to_datetime(data.index, errors='coerce')
                        data = data[data.index.notnull()]
                        data = data.sort_index()
                        data = data.apply(pd.to_numeric, errors='coerce')
                        
                        st.session_state['prices_full'] = data
                        st.session_state['volume_full'] = None 
                        st.session_state['benchmark_data'] = None
                        st.session_state['current_benchmark'] = None
                        st.session_state['last_uploaded_file'] = file_hash
                        
                    except Exception as e:
                        st.error(f"Incompatibilidad estructural de lectura: {e}")
            
            if st.session_state.get('prices_full') is not None:
                st.success(f"Base de datos operativa: {uploaded_file.name}")

    if st.session_state.get('prices_full') is not None:
        prices_full = st.session_state['prices_full']
        current_bench = st.session_state.get('current_benchmark')

        if current_bench != benchmark_ticker or st.session_state.get('benchmark_data') is None:
            with st.spinner(f"Sincronizando vector del Benchmark ({benchmark_ticker})..."):
                try:
                    min_d = prices_full.index.min()
                    max_d = prices_full.index.max()
                    max_d_yf = max_d + timedelta(days=1)
                    
                    new_bench_raw = yf.download(benchmark_ticker, start=min_d, end=max_d_yf, auto_adjust=True, progress=False)
                    
                    if isinstance(new_bench_raw.columns, pd.MultiIndex):
                        new_bench = new_bench_raw['Close'].copy()
                    elif 'Close' in new_bench_raw.columns:
                        new_bench = new_bench_raw[['Close']].copy()
                    else:
                        new_bench = new_bench_raw.copy()
                        
                    if isinstance(new_bench, pd.Series):
                        new_bench = new_bench.to_frame()
                    new_bench.columns = [benchmark_ticker]
                    
                    st.session_state['benchmark_data'] = new_bench
                    st.session_state['current_benchmark'] = benchmark_ticker
                except Exception as e:
                    st.warning(f"Error de sincronización con Benchmark: {e}")

    if st.session_state.get('prices_full') is not None:
        prices_full = st.session_state['prices_full']

        st.divider()
        st.subheader("2. Filtro de Período de Análisis")

        if prices_full.empty or pd.isna(prices_full.index.min()):
            st.error("No se detectaron registros válidos en la base de datos para este intervalo. Se requiere seleccionar una frecuencia menor (Mensual o Diaria) o ampliar el rango de fechas.")
            st.stop()

        min_date = prices_full.index.min().date()
        max_date = prices_full.index.max().date()

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_start = st.date_input("Fecha de inicio", value=min_date, min_value=min_date, max_value=max_date, key="filt_start_m1")
        with col_f2:
            filter_end = st.date_input("Fecha de fin", value=max_date, min_value=min_date, max_value=max_date, key="filt_end_m1")

        prices = prices_full.loc[filter_start:filter_end]
        
        benchmark_data = st.session_state.get('benchmark_data')
        benchmark_filtered = None
        if benchmark_data is not None:
            try:
                benchmark_filtered = benchmark_data.loc[filter_start:filter_end]
            except:
                benchmark_filtered = None

        if len(prices) == 0:
            st.warning("No se detectan registros en el período especificado.")
        else:
            st.subheader("3. Selección de Activos")

            if 'selected_assets_m1' not in st.session_state:
                st.session_state.selected_assets_m1 = prices.columns.tolist()

            valid_selected = [t for t in st.session_state.selected_assets_m1 if t in prices.columns.tolist()]

            selected_assets = st.multiselect(
                "Instrumentos a procesar en el módulo",
                options=prices.columns.tolist(),
                default=valid_selected if valid_selected else prices.columns.tolist(),
                key="multi_assets_m1"
            )
            st.session_state.selected_assets_m1 = selected_assets

            if len(selected_assets) == 0:
                st.warning("Es mandatorio seleccionar al menos un instrumento financiero.")
            else:
                prices_selected = prices[selected_assets]
                prices_selected = prices_selected.dropna(axis=1, how='all')
                returns = prices_selected.pct_change().dropna() 
                if len(returns) == 0:
                    st.error("Error crítico:No existe un período de tiempo superpuesto con datos válidos para TODOS los activos seleccionados. Verifica que no haya tickers inválidos o activos que no cotizaban en estas fechas.")
                    st.stop() 
                
                
                rf_rate_m1 = get_dynamic_rf(prices_selected)
                if usar_ticker_rf:
                    st.info(f"Tasa Libre de Riesgo operativa ({rf_ticker}): **{rf_rate_m1*100:.2f}%**")
                else:
                    st.info(f"Tasa Libre de Riesgo operativa (Tasa Fija): **{rf_rate_m1*100:.2f}%**")

                st.subheader("4. Vista Previa de Base de Datos")
                st.dataframe(prices_selected.tail(10), use_container_width=True)

                st.subheader("5. Métricas de Riesgo y Rentabilidad (Anualizadas)")

                arithmetic_return = (1 + returns.mean()) ** ann_factor - 1
                geo_return = (prices_selected.iloc[-1] / prices_selected.iloc[0]) ** (ann_factor / len(returns)) - 1
                annual_vol = returns.std() * np.sqrt(ann_factor)
                sharpe_ratio = (arithmetic_return - rf_rate_m1) / annual_vol

                cumulative = (1 + returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()

                downside_returns = returns[returns < 0]
                downside_std = downside_returns.std() * np.sqrt(ann_factor)
                sortino_ratio = (arithmetic_return - rf_rate_m1) / downside_std

                summary = pd.DataFrame({
                    'Retorno Anualizado Aritmético (%)': (arithmetic_return * 100).round(2),
                    'Retorno Anualizado Geométrico (%)': (geo_return * 100).round(2),
                    'Volatilidad Anualizada (%)': (annual_vol * 100).round(2),
                    'Ratio de Sharpe': sharpe_ratio.round(3),
                    'Ratio de Sortino': sortino_ratio.round(3),
                    'Maximum Drawdown (%)': (max_drawdown * 100).round(2),
                    'Precio Actual': prices_selected.iloc[-1].round(2)
                }, index=selected_assets)

                st.dataframe(summary, use_container_width=True)
                st.caption(f"**Base Metodológica:** Anualización sustentada en un factor temporal de {ann_factor} períodos.")
                
                st.markdown("""
                **Interpretación Institucional de Ratios:**
                * **Sharpe:** Rendimiento excedente generado por cada unidad de riesgo sistemático. 
                * **Sortino:** Penaliza exclusivamente la volatilidad a la baja (pérdidas), ignorando la varianza positiva.
                * **Maximum Drawdown:** Riesgo de cola extrema; ilustra la pérdida máxima de capital desde un máximo histórico.
                """)

                st.subheader("6. Estadísticos Descriptivos Avanzados")

                mean_periodic = (returns.mean() * 100).round(4)
                std_periodic = (returns.std() * 100).round(4)
                skewness = returns.skew().round(4)
                kurtosis = returns.kurtosis().round(4)

                jb_results = returns.apply(lambda x: jarque_bera(x))
                jb_stat = jb_results.apply(lambda x: round(x[0], 2))
                jb_pvalue = jb_results.apply(lambda x: round(x[1], 4))

                var_95 = (returns.quantile(0.05) * 100).round(4)
                var_99 = (returns.quantile(0.01) * 100).round(4)
                cvar_95 = returns.apply(lambda x: x[x <= x.quantile(0.05)].mean() * 100).round(4)
                cvar_99 = returns.apply(lambda x: x[x <= x.quantile(0.01)].mean() * 100).round(4)

                stats_df = pd.DataFrame({
                    'Media Periódica (%)': mean_periodic,
                    'Desviación Estándar Periódica (%)': std_periodic,
                    'Asimetría (Skewness)': skewness,
                    'Curtosis (Kurtosis)': kurtosis,
                    'Jarque-Bera (Estadístico)': jb_stat,
                    'Jarque-Bera (p-value)': jb_pvalue,
                    'VaR Histórico 95% (1 período, %)': var_95,
                    'VaR Histórico 99% (1 período, %)': var_99,
                    'CVaR Histórico 95% (1 período, %)': cvar_95,
                    'CVaR Histórico 99% (1 período, %)': cvar_99
                }).T
                stats_df.index.name = 'Estadístico'
                stats_df.columns.name = 'Activo'

                st.dataframe(stats_df, use_container_width=True)
                
                stats_pdf_df = stats_df.T.reset_index().rename(columns={'index':'Activo'})
                st.session_state['pdf_tbl_stats'] = create_pdf_table(stats_pdf_df)

                # ============================================
                # 7. CLASIFICACIÓN SECTORIAL
                # ============================================
                st.subheader("7. Clasificación Sectorial de Activos")
                
                sector_etfs = {
                    'XLK': 'Tecnología', 'XLF': 'Financieros', 'XLC': 'Comunicaciones',
                    'XLY': 'Consumo Discrecional', 'XLI': 'Industriales', 'XLV': 'Salud',
                    'XLP': 'Consumo Básico', 'XLE': 'Energía', 'XLU': 'Utilidades',
                    'XLB': 'Materiales', 'XLRE': 'Bienes Raíces'
                }

                modo_clasif = st.radio("Metodología de Clasificación", ["Automática por Correlación Cruzada (ETFs SPDR)", "Asignación Manual"], horizontal=True, key="rad_sector_mode")

                if 'sector_df' not in st.session_state:
                    st.session_state.sector_df = pd.DataFrame({'Activo': selected_assets, 'Sector': ['General'] * len(selected_assets)})
                else:
                    current_assets = set(selected_assets)
                    saved_assets = set(st.session_state.sector_df['Activo'])
                    if current_assets != saved_assets:
                        new_rows = [{'Activo': a, 'Sector': 'General'} for a in current_assets - saved_assets]
                        filtered_df = st.session_state.sector_df[st.session_state.sector_df['Activo'].isin(current_assets)]
                        if new_rows:
                            st.session_state.sector_df = pd.concat([filtered_df, pd.DataFrame(new_rows)], ignore_index=True)
                        else:
                            st.session_state.sector_df = filtered_df

                if modo_clasif == "Automática por Correlación Cruzada (ETFs SPDR)":
                    if st.button("Ejecutar Autoclasificación", type="primary", key="btn_auto_sec"):
                        with st.spinner("Calculando proximidad matricial contra índices sectoriales..."):
                            try:
                                etf_tickers = list(sector_etfs.keys())
                                end_f_yf = filter_end + timedelta(days=1)
                                etf_data = yf.download(etf_tickers, start=filter_start, end=end_f_yf, auto_adjust=True, progress=False)['Close']
                                etf_returns = etf_data.pct_change().dropna()

                                common_idx = returns.index.intersection(etf_returns.index)
                                user_rets_aligned = returns.loc[common_idx]
                                etf_rets_aligned = etf_returns.loc[common_idx]

                                new_classifications = []
                                for asset in selected_assets:
                                    corrs = etf_rets_aligned.apply(lambda col: user_rets_aligned[asset].corr(col))
                                    best_etf = corrs.idxmax()
                                    new_classifications.append({'Activo': asset, 'Sector': sector_etfs[best_etf]})

                                st.session_state.sector_df = pd.DataFrame(new_classifications)
                            except Exception as e:
                                st.error(f"Falla algorítmica: {e}")

                edited_sectors = st.data_editor(st.session_state.sector_df, use_container_width=True, hide_index=True, num_rows="fixed", key="dt_editor_sector")
                st.session_state.sector_df = edited_sectors

                st.subheader("8. Relación Riesgo-Rentabilidad Sectorial")
                
                summary_with_sector = summary.reset_index().rename(columns={'index': 'Activo'})
                summary_with_sector = summary_with_sector.merge(edited_sectors, on='Activo', how='left')

                fig_scatter = px.scatter(
                    summary_with_sector,
                    x='Volatilidad Anualizada (%)', y='Retorno Anualizado Aritmético (%)',
                    color='Sector',
                    text='Activo', hover_name='Activo',
                    title="Posicionamiento de Dispersión Bivariada"
                )
                fig_scatter.update_traces(textposition='top center', marker=dict(size=10))
                fig_scatter.update_layout(height=520, plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#E5E5E5'), yaxis=dict(showgrid=True, gridcolor='#E5E5E5'))
                st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_m1")

                st.subheader("9. Serie de Retornos (HPR)")
                st.dataframe(returns.round(6).tail(10), use_container_width=True)

                st.subheader("10. Exportación de Datos")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Descargar Vector de Precios (CSV)", data=prices_selected.to_csv().encode('utf-8'), file_name="precios.csv", mime="text/csv", key="dl_px_m1")
                with col2:
                    st.download_button("Descargar Matriz de Retornos (CSV)", data=returns.to_csv().encode('utf-8'), file_name="retornos.csv", mime="text/csv", key="dl_ret_m1")

                st.subheader("11. Heatmap de Retornos Mensuales Compuestos")
                monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                monthly_pivot = monthly_returns.copy()
                monthly_pivot['Year'] = monthly_pivot.index.year
                monthly_pivot['Month'] = monthly_pivot.index.month

                selected_ticker_heatmap = st.selectbox("Instrumento a evaluar:", options=selected_assets, index=0, key="sel_heat_m1")
                pivot_table = monthly_pivot.pivot_table(values=selected_ticker_heatmap, index='Year', columns='Month', aggfunc='mean')
                month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                pivot_table.columns = month_names[:len(pivot_table.columns)]

                fig_heat, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(pivot_table * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
                ax.set_title(f"Retornos Históricos Mensuales - {selected_ticker_heatmap} (%)", fontsize=13)
                st.pyplot(fig_heat)
                plt.close(fig_heat)

                st.subheader("12. Evolución del Precio (Base 100)")
                
                opciones_grafico = ["Todos los activos seleccionados"] + selected_assets
                activo_visualizar = st.selectbox("Seleccionar instrumento para graficar junto al Benchmark:", options=opciones_grafico, key="sel_line_m1")
                
                if activo_visualizar == "Todos los activos seleccionados":
                    activos_grafica_linea = selected_assets
                    titulo_linea = "Crecimiento de Capital Interactivo (Base 100)"
                else:
                    activos_grafica_linea = [activo_visualizar]
                    titulo_linea = f"Crecimiento de Capital (Base 100) - {activo_visualizar}"
                
                normalized = (prices_selected[activos_grafica_linea] / prices_selected[activos_grafica_linea].iloc[0]) * 100
                fig_line = px.line(normalized, title=titulo_linea)

                if benchmark_filtered is not None and not benchmark_filtered.empty:
                    try:
                        if benchmark_ticker in benchmark_filtered.columns:
                            bench_s = benchmark_filtered[benchmark_ticker].dropna()
                            if len(bench_s) > 0:
                                bench_norm = (bench_s / bench_s.iloc[0]) * 100
                                fig_line.add_scatter(
                                    x=bench_norm.index, 
                                    y=bench_norm, 
                                    mode='lines', 
                                    name=f"Benchmark ({benchmark_ticker})", 
                                    line=dict(dash='dash', color='#002060', width=2.5)
                                )
                    except:
                        pass

                fig_line.update_layout(xaxis_rangeslider_visible=True, height=550, plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#E5E5E5'), yaxis=dict(showgrid=True, gridcolor='#E5E5E5', title="Valor Base 100"))
                st.plotly_chart(fig_line, use_container_width=True, key="line_m1")
                st.session_state['m1_fig_line'] = fig_line

                if st.session_state.get('volume_full') is not None:
                    st.subheader("13. Volumen de Transacciones Operadas")
                    vol_filtered = st.session_state['volume_full'].loc[filter_start:filter_end][selected_assets]
                    vol_ticker = st.selectbox("Seleccionar instrumento base:", options=selected_assets, key="sel_vol_m1")
                    fig_vol = px.bar(vol_filtered, x=vol_filtered.index, y=vol_ticker, title=f"Distribución de Volumen - {vol_ticker}")
                    fig_vol.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#E5E5E5'), yaxis=dict(showgrid=True, gridcolor='#E5E5E5', title="Volumen"))
                    fig_vol.update_traces(marker_color='#D4AF37')
                    st.plotly_chart(fig_vol, use_container_width=True, key="vol_m1")

# ============================================
# MÓDULO 2: OPTIMIZACIÓN DE PORTAFOLIOS
# ============================================
with tab2:
    st.header("Módulo 2: Optimización de Portafolios")

    if 'prices_full' not in st.session_state or st.session_state['prices_full'] is None:
        st.warning("Es requerido cargar datos en el Módulo 1 antes de proceder con la optimización.")
    else:
        prices_full = st.session_state['prices_full']
        f_start = st.session_state.get('filt_start_m1', prices_full.index.min().date())
        f_end = st.session_state.get('filt_end_m1', prices_full.index.max().date())

        st.subheader("1. Selección de Activos para Optimización")
        
        default_assets = st.session_state.get('selected_assets_m1', [])
        valid_default = [a for a in default_assets if a in prices_full.columns]

        selected_assets_m2 = st.multiselect(
            "Instrumentos a procesar en el optimizador matricial",
            options=prices_full.columns.tolist(),
            default=valid_default if valid_default else prices_full.columns.tolist(),
            key="multi_assets_m2"
        )
        
        if len(selected_assets_m2) < 2:
            st.warning("Se requiere la selección de al menos 2 activos para ejecutar la optimización de portafolios.")
        else:
            prices_m2 = prices_full.loc[f_start:f_end][selected_assets_m2]
            returns_m2 = prices_m2.pct_change().dropna()
            num_activos = len(selected_assets_m2)
            
            # ============================================
            # 2. PARÁMETROS DE RESTRICCIÓN
            # ============================================
            st.subheader("2. Parámetros de Restricción Matemática")
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                min_peso_posible = float(1.0 / num_activos)
                max_weight = st.slider(
                    "Peso máximo permitido por activo (%)",
                    min_value=round(min_peso_posible * 100, 1),
                    max_value=100.0,
                    value=100.0,
                    step=1.0,
                    help="Establece un límite superior para mitigar soluciones de esquina y evitar concentración extrema.",
                    key="sld_max_weight_m2"
                ) / 100.0
            
            with col_r2:
                st.markdown("<br>", unsafe_allow_html=True)
                allow_shorts = st.checkbox(
                    "Permitir ventas en corto (Short Selling)",
                    value=False,
                    help="Habilita ponderaciones negativas hasta un límite del -50% por activo, expandiendo la frontera eficiente de manera realista.",
                    key="chk_shorts_m2"
                )

            # ============================================
            # 3. CARACTERIZACIÓN DE MERCADO Y LIQUIDEZ
            # ============================================
            st.subheader("3. Caracterización de Mercado y Filtros de Liquidez")
            st.markdown("Definición de las condiciones de mercado del universo seleccionado.")

            if 'market_data_df_m2' not in st.session_state or set(st.session_state.market_data_df_m2['Activo']) != set(selected_assets_m2):
                init_rows = []
                for asset in selected_assets_m2:
                    init_rows.append({
                        'Activo': asset, 
                        'Capitalización de Mercado (Millones USD)': 50000.0, 
                        'Bid-Ask Spread (%)': 0.05
                    })
                st.session_state.market_data_df_m2 = pd.DataFrame(init_rows)

            if data_source == "Descargar de Yahoo Finance":
                if st.button("Consultar datos de mercado en vivo (Yahoo Finance)", key="btn_fetch_mcap"):
                    with st.spinner("Extrayendo capitalización bursátil y spreads de liquidez..."):
                        updated_rows = []
                        for asset in selected_assets_m2:
                            try:
                                t = yf.Ticker(asset)
                                info = t.info
                                mcap = info.get('marketCap') or info.get('totalAssets') or 50000000000
                                mcap = mcap / 1000000.0
                                spread = info.get('bidAskSpread', 0.05)
                                if not isinstance(spread, (int, float)) or spread == 0:
                                    spread = 0.05
                                updated_rows.append({
                                    'Activo': asset, 
                                    'Capitalización de Mercado (Millones USD)': round(mcap, 2), 
                                    'Bid-Ask Spread (%)': round(spread, 3)
                                })
                            except:
                                updated_rows.append({
                                    'Activo': asset, 
                                    'Capitalización de Mercado (Millones USD)': 50000.0, 
                                    'Bid-Ask Spread (%)': 0.05
                                })
                        st.session_state.market_data_df_m2 = pd.DataFrame(updated_rows)

            edited_market_data = st.data_editor(st.session_state.market_data_df_m2, use_container_width=True, hide_index=True, key="dt_editor_market_m2")
            st.session_state.market_data_df_m2 = edited_market_data

            # ============================================
            # 4. DATOS DE ENTRADA PARA OPTIMIZACIÓN
            # ============================================
            mean_returns = returns_m2.mean() * ann_factor
            cov_matrix = returns_m2.cov() * ann_factor
            corr_matrix = returns_m2.corr()
            asset_vols = returns_m2.std() * np.sqrt(ann_factor)
            rf_rate_m2 = get_dynamic_rf(prices_m2)

            st.subheader("4. Datos de Entrada para la Optimización")
            
            input_data = pd.DataFrame({
                'Activo': selected_assets_m2,
                'Retorno Esperado Anualizado (%)': (mean_returns.values * 100).round(2),
                'Volatilidad Anualizada (%)': (asset_vols.values * 100).round(2)
            })
            input_data = input_data.merge(edited_market_data, on='Activo', how='left')
            st.dataframe(input_data, use_container_width=True, hide_index=True)

            # ============================================
            # 5. MATRICES EN PESTAÑAS (OPTIMIZACIÓN DE ESPACIO)
            # ============================================
            st.subheader("5. Matrices de Correlación y Covarianza")
            
            subtab_corr, subtab_cov = st.tabs(["Matriz de Correlación de Pearson", "Matriz de Covarianza (Anualizada)"])
            
            with subtab_corr:
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=".2f", 
                    color_continuous_scale='RdBu_r',
                    title="Interconexión de Activos (Correlación)",
                    zmin=-1, zmax=1
                )
                fig_corr.update_layout(height=550, plot_bgcolor='white')
                st.plotly_chart(fig_corr, use_container_width=True, key="mat_corr_m2")
            
            with subtab_cov:
                fig_cov = px.imshow(
                    cov_matrix, 
                    text_auto=".4f", 
                    color_continuous_scale='YlOrRd',
                    title="Estructura de Variabilidad Colectiva (Covarianza)"
                )
                fig_cov.update_layout(height=550, plot_bgcolor='white')
                st.plotly_chart(fig_cov, use_container_width=True, key="mat_cov_m2")

            # ============================================
            # PROCESAMIENTO MATRICIAL Y OPTIMIZACIÓN
            # ============================================
            def portfolio_performance(weights, mean_returns, cov_matrix):
                p_ret = np.sum(mean_returns * weights)
                p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return p_ret, p_vol

            def min_variance_objective(weights, mean_returns, cov_matrix):
                return portfolio_performance(weights, mean_returns, cov_matrix)[1]

            def max_sharpe_objective(weights, mean_returns, cov_matrix, rf_rate):
                p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
                return -(p_ret - rf_rate) / p_vol

            def calculate_dr_metrics(weights, asset_vols, portfolio_vol):
                weighted_vol = np.sum(np.abs(weights) * asset_vols)
                dr = weighted_vol / portfolio_vol
                db = 1.0 - (1.0 / dr)
                return dr, db

            lower_bound = -0.5 if allow_shorts else 0.0
            bounds = [(lower_bound, max_weight) for _ in range(num_activos)]
            cons_base = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

            res_gmv = minimize(min_variance_objective, num_activos * [1.0 / num_activos], args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=cons_base)
            w_gmv = res_gmv.x
            ret_gmv, vol_gmv = portfolio_performance(w_gmv, mean_returns, cov_matrix)
            sharpe_gmv = (ret_gmv - rf_rate_m2) / vol_gmv

            res_ms = minimize(max_sharpe_objective, num_activos * [1.0 / num_activos], args=(mean_returns, cov_matrix, rf_rate_m2), method='SLSQP', bounds=bounds, constraints=cons_base)
            w_ms = res_ms.x
            ret_ms, vol_ms = portfolio_performance(w_ms, mean_returns, cov_matrix)
            sharpe_ms = (ret_ms - rf_rate_m2) / vol_ms

            w_ew = np.ones(num_activos) / num_activos
            ret_ew, vol_ew = portfolio_performance(w_ew, mean_returns, cov_matrix)
            sharpe_ew = (ret_ew - rf_rate_m2) / vol_ew

            dr_gmv, db_gmv = calculate_dr_metrics(w_gmv, asset_vols, vol_gmv)
            dr_ms, db_ms = calculate_dr_metrics(w_ms, asset_vols, vol_ms)
            dr_ew, db_ew = calculate_dr_metrics(w_ew, asset_vols, vol_ew)

            # ============================================
            # 6. TABLA COMPARATIVA DE PORTAFOLIOS
            # ============================================
            st.subheader("6. Tabla Comparativa de Portafolios Óptimos")

            comparison_df = pd.DataFrame({
                'Métrica de Rendimiento y Riesgo': [
                    'Retorno Esperado Anualizado (%)',
                    'Volatilidad Anualizada (%)',
                    'Ratio de Sharpe',
                    'Ratio de Diversificación (DR)',
                    'Beneficio de Diversificación (%)'
                ],
                'Mínima Varianza': [
                    round(ret_gmv * 100, 2),
                    round(vol_gmv * 100, 2),
                    round(sharpe_gmv, 3),
                    round(dr_gmv, 3),
                    round(db_gmv * 100, 2)
                ],
                'Máximo Sharpe (Tangente)': [
                    round(ret_ms * 100, 2),
                    round(vol_ms * 100, 2),
                    round(sharpe_ms, 3),
                    round(dr_ms, 3),
                    round(db_ms * 100, 2)
                ],
                'Equiponderado (1/N)': [
                    round(ret_ew * 100, 2),
                    round(vol_ew * 100, 2),
                    round(sharpe_ew, 3),
                    round(dr_ew, 3),
                    round(db_ew * 100, 2)
                ]
            })

            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.caption("""
            **Interpretación Metodológica (Beneficio de Diversificación):** El valor porcentual ilustra la fracción de riesgo individual que ha sido mitigada gracias a las covarianzas de la cartera. 
            Un beneficio de 8.4% significa que la volatilidad del portafolio es un 8.4% inferior a la suma ponderada de los riesgos de los activos aislados.
            """)

            def get_dr_interpretation(dr_val):
                if dr_val <= 1.05:
                    return "Concentración Absoluta (Beneficio nulo o marginal)."
                elif 1.05 < dr_val <= 1.3:
                    return "Diversificación Baja (Concentración direccional o sectorial)."
                elif 1.3 < dr_val <= 2.0:
                    return "Diversificación Institucional Óptima (Estructura matricial eficiente)."
                else:
                    return "Diversificación Extrema / Cobertura (Alta descorrelación)."

            st.markdown("**Evaluación Estructural de Diversificación (Semaforización):**")
            st.info(f"""
            * **Mínima Varianza:** DR = {dr_gmv:.3f} : {get_dr_interpretation(dr_gmv)}
            * **Máximo Sharpe:** DR = {dr_ms:.3f} : {get_dr_interpretation(dr_ms)}
            * **Equiponderado (1/N):** DR = {dr_ew:.3f} : {get_dr_interpretation(dr_ew)}
            """)

            # ============================================
            # 7. FRONTERA EFICIENTE CON ACTIVOS DISPERSOS
            # ============================================
            st.subheader("7. Frontera Eficiente de Markowitz y Capital Market Line (CML)")
            
            with st.spinner("Construyendo espacio de búsqueda y optimización de frontera..."):
                num_portfolios_sim = 1500
                sim_results = np.zeros((3, num_portfolios_sim))
                for i in range(num_portfolios_sim):
                    w = np.random.normal(0.5, 0.3, num_activos) if allow_shorts else np.random.random(num_activos)
                    w /= np.sum(w)
                    w = np.clip(w, lower_bound, max_weight)
                    w /= np.sum(w)

                    p_ret, p_vol = portfolio_performance(w, mean_returns, cov_matrix)
                    sim_results[0, i] = p_vol
                    sim_results[1, i] = p_ret
                    sim_results[2, i] = (p_ret - rf_rate_m2) / p_vol

                target_returns = np.linspace(mean_returns.min() * (1.2 if allow_shorts else 1), max(mean_returns.max() * (1.5 if allow_shorts else 1), ret_ms * 1.2), 40)
                frontier_vols = []
                for tr in target_returns:
                    cons_frontier = (
                        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                        {'type': 'eq', 'fun': lambda w: np.sum(mean_returns * w) - tr}
                    )
                    res_f = minimize(min_variance_objective, num_activos * [1.0 / num_activos], args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=cons_frontier)
                    if res_f.success and res_f.fun < (vol_ms * 3.0):
                        frontier_vols.append(res_f.fun)
                    else:
                        frontier_vols.append(None)

                valid_frontier = [(v, r) for v, r in zip(frontier_vols, target_returns) if v is not None]
                if valid_frontier:
                    frontier_vols_valid = [x[0] for x in valid_frontier]
                    frontier_rets_valid = [x[1] for x in valid_frontier]
                else:
                    frontier_vols_valid, frontier_rets_valid = [], []

                cml_vols = np.linspace(0, max([v for v in frontier_vols_valid] + [vol_ms]) * 1.2, 20)
                cml_rets = rf_rate_m2 + cml_vols * sharpe_ms

            fig_frontier = go.Figure()

            # Fondo del espacio de búsqueda
            fig_frontier.add_trace(go.Scatter(
                x=sim_results[0]*100, y=sim_results[1]*100, mode='markers',
                marker=dict(color=sim_results[2], colorscale='Viridis', showscale=True, colorbar=dict(title="Ratio de Sharpe")),
                name='Espacio de Búsqueda',
                hoverinfo='text',
                text=[f"Retorno: {r:.2f}%<br>Volatilidad: {v:.2f}%<br>Sharpe: {s:.3f}" for r, v, s in zip(sim_results[1]*100, sim_results[0]*100, sim_results[2])]
            ))

            # Curva de la frontera eficiente
            if frontier_vols_valid:
                fig_frontier.add_trace(go.Scatter(
                    x=[v*100 for v in frontier_vols_valid], y=[r*100 for r in frontier_rets_valid],
                    mode='lines', line=dict(color='#D4AF37', width=3), name='Frontera Eficiente (Markowitz)'
                ))

            # Capital Market Line
            fig_frontier.add_trace(go.Scatter(
                x=[v*100 for v in cml_vols], y=[r*100 for r in cml_rets],
                mode='lines', line=dict(color='#002060', width=2.5, dash='dash'), name='Capital Market Line (CML)'
            ))

            # Activos Individuales Dispersos
            fig_frontier.add_trace(go.Scatter(
                x=[v*100 for v in asset_vols], y=[r*100 for r in mean_returns],
                mode='markers+text',
                marker=dict(color='#4A5568', size=8, symbol='circle', line=dict(color='white', width=1)),
                text=selected_assets_m2, textposition='top center', name='Activos Individuales'
            ))

            # Portafolios Clave
            fig_frontier.add_trace(go.Scatter(
                x=[vol_gmv*100], y=[ret_gmv*100], mode='markers+text',
                marker=dict(color='red', size=12, symbol='diamond'),
                text=['Mínima Varianza'], textposition='top left', name='Mínima Varianza'
            ))

            fig_frontier.add_trace(go.Scatter(
                x=[vol_ms*100], y=[ret_ms*100], mode='markers+text',
                marker=dict(color='gold', size=14, symbol='star', line=dict(color='black', width=1)),
                text=['Máximo Sharpe (Tangente)'], textposition='top right', name='Máximo Sharpe'
            ))

            fig_frontier.add_trace(go.Scatter(
                x=[vol_ew*100], y=[ret_ew*100], mode='markers+text',
                marker=dict(color='orange', size=12, symbol='square'),
                text=['Equiponderado (1/N)'], textposition='bottom right', name='Equiponderado'
            ))

            max_vol_plot = max([vol_ms * 2.0] + [vol_ew * 1.5] + [asset_vols.max()]) * 100
            
            fig_frontier.update_layout(
                xaxis=dict(title='Volatilidad Anualizada (%)', showgrid=True, gridcolor='#E5E5E5', range=[0, max_vol_plot]),
                yaxis=dict(title='Retorno Esperado Anualizado (%)', showgrid=True, gridcolor='#E5E5E5'),
                plot_bgcolor='white', height=600,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_frontier, use_container_width=True, key="plotly_frontier_m2")

            # ============================================
            # 8. CARTERA COMPLETA (AVERSIÓN AL RIESGO)
            # ============================================
            st.subheader("8. Asignación de Cartera Completa (Riesgoso + Libre de Riesgo)")
            st.markdown("Integración del Portafolio Tangente con el Activo Libre de Riesgo en función del perfil del inversionista.")
            
            col_calc1, col_calc2 = st.columns(2)
            with col_calc1:
                calc_mode = st.radio("Metodología de Asignación:", ["Por Coeficiente de Aversión al Riesgo (A)", "Definir porcentaje (y) manualmente"], key="rad_calc_m2")
            
            with col_calc2:
                if calc_mode == "Por Coeficiente de Aversión al Riesgo (A)":
                    A_coef = st.number_input("Coeficiente de Aversión al Riesgo (A)", min_value=1.0, max_value=20.0, value=3.0, step=0.5, key="num_A_coef")
                    y_opt = (ret_ms - rf_rate_m2) / (A_coef * (vol_ms ** 2))
                else:
                    y_opt = st.slider("Proporción del capital en Portafolio Tangente (y) %", min_value=0.0, max_value=200.0, value=100.0, step=1.0, key="sld_y_opt") / 100.0

            weight_rf = 1.0 - y_opt
            comp_ret = rf_rate_m2 + y_opt * (ret_ms - rf_rate_m2)
            comp_vol = y_opt * vol_ms

            st.info(f"""
            **Métricas de la Cartera Completa Estructurada:**
            * Retorno Esperado Anualizado: **{(comp_ret * 100):.2f}%**
            * Volatilidad Anualizada: **{(comp_vol * 100):.2f}%**
            * Proporción en Activo Libre de Riesgo: **{(weight_rf * 100):.2f}%**
            * Proporción en Portafolio Riesgoso (Tangente): **{(y_opt * 100):.2f}%**
            """)

            comp_weights = w_ms * y_opt
            comp_weights_df = pd.DataFrame({
                'Activo': selected_assets_m2 + [f"Activo Libre de Riesgo"],
                'Peso en Cartera Completa (%)': np.append(comp_weights * 100, weight_rf * 100).round(2)
            })

            col_ct1, col_ct2 = st.columns([1, 2])
            with col_ct1:
                st.dataframe(comp_weights_df, use_container_width=True, hide_index=True)
            with col_ct2:
                df_pie_comp = comp_weights_df[comp_weights_df['Peso en Cartera Completa (%)'] > 0]
                fig_pie_comp = px.pie(df_pie_comp, values='Peso en Cartera Completa (%)', names='Activo', title='Distribución (Posiciones en Largo)', hole=0.3)
                fig_pie_comp.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie_comp, use_container_width=True, key="pie_comp_m2")

            # ============================================
            # 9. DISTRIBUCIÓN TRANSVERSAL
            # ============================================
            st.subheader("9. Distribución Transversal de Capital por Estrategia Base")

            weights_df = pd.DataFrame({
                'Activo': selected_assets_m2,
                'Mínima Varianza (%)': (w_gmv * 100).round(2),
                'Máximo Sharpe (%)': (w_ms * 100).round(2),
                'Equiponderado (%)': (w_ew * 100).round(2)
            })

            st.dataframe(weights_df, use_container_width=True, hide_index=True)

            weights_df_melted = weights_df.melt(id_vars='Activo', var_name='Estrategia', value_name='Peso (%)')
            weights_df_melted['Activo'] = weights_df_melted['Activo'].astype(str)

            fig_weights = px.bar(
                weights_df_melted, 
                x='Activo', 
                y='Peso (%)', 
                color='Estrategia',
                barmode='group', 
                title='Distribución Transversal de Capital por Estrategia',
                color_discrete_sequence=['#002060', '#63b3ed', '#e53e3e']
            )
            fig_weights.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#E5E5E5', type='category', title="Activo"),
                yaxis=dict(title='Peso Asignado (%)', showgrid=True, gridcolor='#E5E5E5'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_weights, use_container_width=True, key="plotly_weights_bar_m2")

            st.markdown("**Desglose de Composición Estructural (Donas)**")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                df_pie_gmv = weights_df[weights_df['Mínima Varianza (%)'] > 0]
                fig_pie_gmv = px.pie(df_pie_gmv, values='Mínima Varianza (%)', names='Activo', title='Mínima Varianza', hole=0.3)
                fig_pie_gmv.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie_gmv.update_layout(showlegend=False)
                st.plotly_chart(fig_pie_gmv, use_container_width=True, key="pie_gmv_m2")
                
            with col_p2:
                df_pie_ms = weights_df[weights_df['Máximo Sharpe (%)'] > 0]
                fig_pie_ms = px.pie(df_pie_ms, values='Máximo Sharpe (%)', names='Activo', title='Máximo Sharpe', hole=0.3)
                fig_pie_ms.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie_ms.update_layout(showlegend=False)
                st.plotly_chart(fig_pie_ms, use_container_width=True, key="pie_ms_m2")
                
            with col_p3:
                df_pie_ew = weights_df[weights_df['Equiponderado (%)'] > 0]
                fig_pie_ew = px.pie(df_pie_ew, values='Equiponderado (%)', names='Activo', title='Equiponderado', hole=0.3)
                fig_pie_ew.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie_ew.update_layout(showlegend=False)
                st.plotly_chart(fig_pie_ew, use_container_width=True, key="pie_ew_m2")

            # ============================================
            # 10. MONTE CARLO CON FRECUENCIA Y APORTES
            # ============================================
            st.subheader("10. Simulador de Proyecciones Estocásticas (Monte Carlo)")
            st.markdown("Proyección probabilística basada en el modelo de Caminata Aleatoria Geométrica (GBM) incorporando aportes periódicos.")

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                horizon_years = st.number_input("Horizonte temporal de proyección (Años)", min_value=1, max_value=25, value=10, step=1, key="num_horizon_m2")
            with col_s2:
                initial_investment = st.number_input("Capital Inicial Requerido (USD)", min_value=1000, max_value=5000000, value=10000, step=5000, key="num_inv_initial_m2")
            with col_s3:
                aporte_periodico = st.number_input("Aporte Monetario Periódico (USD)", min_value=0, max_value=100000, value=100, step=50, key="num_aporte_m2")

            col_sf1, col_sf2 = st.columns(2)
            with col_sf1:
                portafolio_sim = st.selectbox(
                    "Estrategia base para la simulación:",
                    ["Mínima Varianza", "Máximo Sharpe", "Equiponderado", "Cartera Completa"],
                    key="sel_port_sim_m2"
                )
            with col_sf2:
                freq_sim = st.radio("Frecuencia de simulación y aportes:", ["Mensual", "Anual"], horizontal=True, key="rad_freq_sim_m2")

            if portafolio_sim == "Mínima Varianza":
                mu_sim, sigma_sim = ret_gmv, vol_gmv
            elif portafolio_sim == "Máximo Sharpe":
                mu_sim, sigma_sim = ret_ms, vol_ms
            elif portafolio_sim == "Equiponderado":
                mu_sim, sigma_sim = ret_ew, vol_ew
            else:
                mu_sim, sigma_sim = comp_ret, comp_vol

            if freq_sim == "Mensual":
                steps_per_year = 12
                label_eje_x = "Línea de Tiempo (Meses)"
                total_steps = int(horizon_years * 12)
                time_axis = np.arange(0, total_steps + 1)
            else:
                steps_per_year = 1
                label_eje_x = "Línea de Tiempo (Años)"
                total_steps = int(horizon_years)
                time_axis = np.arange(0, total_steps + 1)

            st.markdown(f"""
            **Resumen de Parámetros del Motor de Proyección ({portafolio_sim}):**
            * Retorno Esperado Anualizado: **{(mu_sim * 100):.2f}%**
            * Volatilidad Anualizada: **{(sigma_sim * 100):.2f}%**
            * Tasa de Deriva de la Frecuencia (Drift per step): **{((mu_sim - 0.5 * (sigma_sim ** 2)) / steps_per_year * 100):.4f}%**
            * Coeficiente de Choque de la Frecuencia (Shock per step): **{((sigma_sim / np.sqrt(steps_per_year)) * 100):.4f}%**
            """)

            if sigma_sim > 0.30:
                st.warning("Nota Técnica: Portafolios con varianza anualizada elevada pueden exhibir distorsiones matemáticas por el efecto de capitalización geométrica continua. Se sugiere basar las decisiones en el percentil 50.")

            num_sims = 1000
            dt = 1.0 / steps_per_year
            drift = (mu_sim - 0.5 * (sigma_sim ** 2)) * dt
            shock = sigma_sim * np.sqrt(dt)

            sim_paths = np.zeros((total_steps + 1, num_sims))
            sim_paths[0, :] = initial_investment

            for t in range(1, total_steps + 1):
                Z = np.random.normal(0, 1, num_sims)
                sim_paths[t, :] = sim_paths[t-1, :] * np.exp(drift + shock * Z) + aporte_periodico

            p5 = np.percentile(sim_paths, 5, axis=1)
            p50 = np.percentile(sim_paths, 50, axis=1)
            p95 = np.percentile(sim_paths, 95, axis=1)

            fig_mc = go.Figure()
            for i in range(60):
                fig_mc.add_trace(go.Scatter(x=time_axis, y=sim_paths[:, i], mode='lines', line=dict(width=0.5, color='rgba(160,160,160,0.2)'), showlegend=False))

            fig_mc.add_trace(go.Scatter(x=time_axis, y=p5, mode='lines', line=dict(color='red', width=2, dash='dash'), name='Percentil 5 (Riesgo de Cola Extrema)'))
            fig_mc.add_trace(go.Scatter(x=time_axis, y=p50, mode='lines', line=dict(color='blue', width=2.5), name='Percentil 50 (Mediana Esperada)'))
            fig_mc.add_trace(go.Scatter(x=time_axis, y=p95, mode='lines', line=dict(color='green', width=2, dash='dash'), name='Percentil 95 (Escenario Optimista)'))

            fig_mc.update_layout(
                title=f"Modelación de Monte Carlo con Aportes: Fondo indexado a {portafolio_sim}",
                xaxis=dict(title=label_eje_x, showgrid=True, gridcolor='#E5E5E5'),
                yaxis=dict(title="Valor Transversal del Capital (USD)", showgrid=True, gridcolor='#E5E5E5'),
                plot_bgcolor='white', height=520,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_mc, use_container_width=True, key="plotly_mc_m2")

            st.markdown(f"**Métricas de Cierre en Horizonte Final:**")
            st.markdown(f"""
            * **Capital Esperado Promedio (Mediana P50):** {p50[-1]:,.2f} USD
            * **Umbral Crítico de Capital (Percentil 5 de Cola):** {p5[-1]:,.2f} USD (Retención de valor en escenarios de severidad con 95% de confianza estadística).
            * **Proyección Excedente Optimista (Percentil 95):** {p95[-1]:,.2f} USD
            """)
# ============================================
# MÓDULO 3: MODELO CAPM Y SECURITY MARKET LINE
# ============================================
with tab3:
    st.header("Módulo 3: Modelo CAPM y Security Market Line (SML)")

    if 'prices_full' not in st.session_state or st.session_state['prices_full'] is None:
        st.warning("Es requerido cargar datos en el Módulo 1 antes de proceder con el análisis CAPM.")
    else:
        prices_full = st.session_state['prices_full']
        f_start = st.session_state.get('filt_start_m1', prices_full.index.min().date())
        f_end = st.session_state.get('filt_end_m1', prices_full.index.max().date())
        
        benchmark_data = st.session_state.get('benchmark_data')
        
        if benchmark_data is None or benchmark_data.empty:
            st.error("No se detectan datos válidos para el índice de referencia (Benchmark). Validar la conexión de red o el archivo de origen.")
        else:
            selected_assets_m3 = st.session_state.get('selected_assets_m2', st.session_state.get('selected_assets_m1', prices_full.columns.tolist()))
            valid_assets_m3 = [a for a in selected_assets_m3 if a in prices_full.columns]
            
            st.subheader("1. Selección de Activos para el Análisis de Sensibilidad")
            selected_assets_m3 = st.multiselect(
                "Instrumentos a evaluar mediante el modelo de valoración de activos:",
                options=prices_full.columns.tolist(),
                default=valid_assets_m3,
                key="multi_assets_m3"
            )
            
            if len(selected_assets_m3) == 0:
                st.warning("Es requerido seleccionar al menos un activo para desplegar las métricas del CAPM.")
            else:
                prices_m3 = prices_full.loc[f_start:f_end][selected_assets_m3]
                returns_m3 = prices_m3.pct_change().dropna()
                
                benchmark_filtered = benchmark_data.loc[f_start:f_end]
                returns_bench = benchmark_filtered.pct_change().dropna()
                
                common_index = returns_m3.index.intersection(returns_bench.index)
                
                if len(common_index) < 5:
                    st.error("Espacio muestral insuficiente en el período filtrado para ejecutar la regresión estadística.")
                else:
                    returns_m3_aligned = returns_m3.loc[common_index]
                    returns_bench_aligned = returns_bench.loc[common_index].iloc[:, 0]
                    
                    rf_rate_m3 = get_dynamic_rf(prices_m3)
                    
                    market_return_annual = (1 + returns_bench_aligned.mean()) ** ann_factor - 1
                    market_risk_premium = market_return_annual - rf_rate_m3
                    
                    st.info(f"""
                    **Parámetros Macroeconómicos del Entorno:**
                    * Tasa Libre de Riesgo ($R_f$): **{rf_rate_m3 * 100:.2f}%**
                    * Rendimiento Anualizado de Mercado ($R_m$): **{market_return_annual * 100:.2f}%**
                    * Prima de Riesgo de Mercado ($R_m - R_f$): **{market_risk_premium * 100:.2f}%**
                    """)
                    
                    # ============================================
                    # ESTIMACIÓN DE REGRESIÓN OLS (BETA Y R2)
                    # ============================================
                    ols_results = {}
                    for asset in selected_assets_m3:
                        slope, intercept, r_value, p_value, std_err = linregress(returns_bench_aligned, returns_m3_aligned[asset])
                        
                        arith_ret = (1 + returns_m3_aligned[asset].mean()) ** ann_factor - 1
                        vol_ann = returns_m3_aligned[asset].std() * np.sqrt(ann_factor)
                        
                        cum_ret = (1 + returns_m3_aligned[asset]).cumprod()
                        max_drawdown = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
                        var_95_periodic = returns_m3_aligned[asset].quantile(0.05)
                        
                        ols_results[asset] = {
                            'Beta': slope,
                            'Intercept': intercept,
                            'R2': r_value ** 2,
                            'Std_Err': std_err,
                            'Retorno_Hist': arith_ret,
                            'Volatilidad': vol_ann,
                            'Max_Drawdown': max_drawdown,
                            'VaR_95': var_95_periodic
                        }
                    
                    st.subheader("2. Regresión Lineal por Mínimos Cuadrados Ordinarios (OLS)")
                    st.markdown("Sensibilidad del rendimiento del activo frente a las fluctuaciones del portafolio de mercado general.")
                    
                    asset_ols_view = st.selectbox("Seleccionar activo para desglosar la regresión bivariada:", options=selected_assets_m3, key="sel_asset_ols")
                    
                    asset_stats = ols_results[asset_ols_view]
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric(label=f"Coeficiente Beta (Sensibilidad β)", value=f"{asset_stats['Beta']:.3f}")
                    with col_m2:
                        st.metric(label=f"Determinación (R²)", value=f"{asset_stats['R2']*100:.2f}%")
                    with col_m3:
                        st.metric(label=f"Error Estándar del Modelo", value=f"{asset_stats['Std_Err']:.4f}")
                        
                    df_scatter_ols = pd.DataFrame({
                        'Market_Returns': returns_bench_aligned * 100,
                        'Asset_Returns': returns_m3_aligned[asset_ols_view] * 100
                    })
                    
                    fig_ols = px.scatter(
                        df_scatter_ols, x='Market_Returns', y='Asset_Returns',
                        labels={'Market_Returns': f'Retornos del Mercado ({benchmark_ticker}) (%)', 'Asset_Returns': f'Retornos de {asset_ols_view} (%)'},
                        title=f"Recta de Regresión Estadística: {asset_ols_view} vs {benchmark_ticker}"
                    )
                    fig_ols.update_traces(marker=dict(color='#002060', opacity=0.6), name='Retornos Periódicos')
                    
                    # Trazo manual de la recta de regresión para garantizar estabilidad
                    x_line = np.array([df_scatter_ols['Market_Returns'].min(), df_scatter_ols['Market_Returns'].max()])
                    y_line = asset_stats['Intercept'] * 100 + asset_stats['Beta'] * x_line
                    
                    fig_ols.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines',
                        line=dict(color='#D4AF37', width=3),
                        name='Recta de Regresión (OLS)'
                    ))
                    
                    fig_ols.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='#E5E5E5'), yaxis=dict(showgrid=True, gridcolor='#E5E5E5'))
                    st.plotly_chart(fig_ols, use_container_width=True, key="plotly_ols_m3")
                    
                    # ============================================
                    # SECURITY MARKET LINE (SML)
                    # ============================================
                    st.subheader("3. Gráfico de la Línea del Mercado de Valores (SML)")
                    st.markdown("Estructura de equilibrio que contrasta el rendimiento histórico compuesto frente al riesgo sistemático (Beta).")
                    
                    betas_list = [ols_results[a]['Beta'] for a in selected_assets_m3]
                    ret_hist_list = [ols_results[a]['Retorno_Hist'] for a in selected_assets_m3]
                    
                    min_beta_line = min(betas_list + [0.0, 1.0]) - 0.2
                    max_beta_line = max(betas_list + [1.0]) + 0.2
                    
                    beta_axis = np.linspace(min_beta_line, max_beta_line, 20)
                    sml_returns = rf_rate_m3 + beta_axis * market_risk_premium
                    
                    fig_sml = go.Figure()
                    
                    fig_sml.add_trace(go.Scatter(
                        x=beta_axis, y=sml_returns * 100,
                        mode='lines', line=dict(color='#002060', width=2.5),
                        name='Security Market Line (SML)'
                    ))
                    
                    fig_sml.add_trace(go.Scatter(
                        x=[1.0], y=[market_return_annual * 100],
                        mode='markers', marker=dict(color='#D4AF37', size=12, symbol='star'),
                        name='Portafolio de Mercado'
                    ))
                    
                    capm_expectations = []
                    jensen_alphas = []
                    valuations = []
                    
                    for asset in selected_assets_m3:
                        b = ols_results[asset]['Beta']
                        r_hist = ols_results[asset]['Retorno_Hist']
                        r_capm = rf_rate_m3 + b * market_risk_premium
                        alpha_j = r_hist - r_capm
                        
                        capm_expectations.append(r_capm)
                        jensen_alphas.append(alpha_j)
                        
                        status = "Subvalorado (Compra / Sobre la SML)" if alpha_j > 0 else "Sobrevalorado (Venta / Bajo la SML)"
                        valuations.append(status)
                        
                        fig_sml.add_trace(go.Scatter(
                            x=[b], y=[r_hist * 100],
                            mode='markers+text',
                            marker=dict(size=10, color='green' if alpha_j > 0 else 'red'),
                            text=[asset], textposition="top center",
                            name=f"{asset}",
                            hoverinfo='text',
                            hovertext=[f"<b>{asset}</b><br>Beta: {b:.3f}<br>Retorno Histórico: {r_hist*100:.2f}%<br>Retorno CAPM: {r_capm*100:.2f}%<br>Alfa de Jensen: {alpha_j*100:.2f}%"],
                            showlegend=False
                        ))
                    
                    fig_sml.update_layout(
                        xaxis=dict(title='Riesgo Sistemático (Coeficiente Beta β)', showgrid=True, gridcolor='#E5E5E5'),
                        yaxis=dict(title='Retorno Anualizado (%)', showgrid=True, gridcolor='#E5E5E5'),
                        plot_bgcolor='white', height=600
                    )
                    st.plotly_chart(fig_sml, use_container_width=True, key="plotly_sml_m3")
                    
                    # ============================================
                    # MATRIZ MATRICIAL RESUMEN CAPM Y RIESGO
                    # ============================================
                    st.subheader("4. Matriz Comparativa y Alfa de Jensen")
                    st.markdown("Consolidación cuantitativa de equilibrio del modelo clásico y caracterización de riesgos de cola históricos.")
                    
                    capm_matrix_df = pd.DataFrame({
                        'Activo': selected_assets_m3,
                        'Beta (β)': [round(ols_results[a]['Beta'], 3) for a in selected_assets_m3],
                        'R² Determinación (%)': [round(ols_results[a]['R2'] * 100, 2) for a in selected_assets_m3],
                        'Rendimiento CAPM (%)': [round(r * 100, 2) for r in capm_expectations],
                        'Rendimiento Histórico (%)': [round(ols_results[a]['Retorno_Hist'] * 100, 2) for a in selected_assets_m3],
                        'Alfa de Jensen (%)': [round(a * 100, 2) for a in jensen_alphas],
                        'Evaluación de Valuación': valuations,
                        'Volatilidad Histórica (%)': [round(ols_results[a]['Volatilidad'] * 100, 2) for a in selected_assets_m3],
                        'Maximum Drawdown (%)': [round(ols_results[a]['Max_Drawdown'] * 100, 2) for a in selected_assets_m3],
                        'VaR Histórico 95% (%)': [round(ols_results[a]['VaR_95'] * 100, 2) for a in selected_assets_m3]
                    })
                    
                    st.dataframe(capm_matrix_df, use_container_width=True, hide_index=True)
                    
                    st.caption("""
                    **Guía para el Análisis Técnico Institucional:**
                    * **Alfa de Jensen > 0 (Subvalorado):** El activo superó el rendimiento requerido por el modelo CAPM dado su nivel de riesgo sistemático. Gráficamente se ubica por encima de la SML, representando una oportunidad de asignación óptima (Generación de Alfa).
                    * **Alfa de Jensen < 0 (Sobrevalorado):** El activo no remunera eficientemente el riesgo asumido. Se posiciona por debajo de la SML, sugiriendo una reducción de exposición o liquidación en carteras eficientes.
                    * **R² (Coeficiente de Determinación):** Mide la proporción de la variabilidad del activo explicada por el mercado general. Un R² alto indica que el riesgo es predominantemente sistemático, validando la precisión de la estimación del CAPM.
                    """)

# ============================================
# CÓDIGO DE EXPORTACIÓN DEL RESUMEN EJECUTIVO (PDF)
# ============================================

if FPDF is not None:
    class ResumenEjecutivoPDF(FPDF):
        def header(self):
            # Franja decorativa superior en Azul Marino Institucional
            self.set_fill_color(0, 32, 96)
            self.rect(0, 0, 210, 8, 'F')
            
            # Línea de acento en Dorado
            self.set_fill_color(212, 175, 55)
            self.rect(0, 8, 210, 1.5, 'F')
            
            # Texto de Encabezado secundario
            self.set_font('Arial', 'B', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 12, self.clean_text('RESUMEN EJECUTIVO DE INVERSIÓN BURSÁTIL | GESTIÓN 2026'), 0, 1, 'R')
            self.ln(4)

        def footer(self):
            # Posicionamiento a 15 mm del borde inferior
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
            
            # Franja decorativa inferior en Dorado
            self.set_fill_color(212, 175, 55)
            self.rect(0, 292, 210, 1.5, 'F')

        def clean_text(self, txt):
            if not isinstance(txt, str):
                txt = str(txt)
            return txt.encode('latin-1', 'replace').decode('latin-1')

        def agregar_titulo_seccion(self, titulo):
            self.set_font('Arial', 'B', 13)
            self.set_text_color(0, 32, 96) 
            self.cell(0, 8, self.clean_text(titulo), 0, 1, 'L')
            
            self.set_fill_color(212, 175, 55)
            self.rect(self.get_x(), self.get_y() + 1, 190, 0.8, 'F')
            self.ln(5)

        def agregar_subtitulo(self, subtitulo):
            self.set_font('Arial', 'B', 10.5)
            self.set_text_color(50, 50, 50)
            self.cell(0, 6, self.clean_text(subtitulo), 0, 1, 'L')
            self.ln(2)


    def generar_pdf_ejecutivo():
        pdf = ResumenEjecutivoPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()
        
        # --------------------------------------------
        # PORTADA Y METADATOS INSTITUCIONALES
        # --------------------------------------------
        pdf.set_font('Arial', 'B', 22)
        pdf.set_text_color(0, 32, 96) 
        pdf.cell(0, 15, pdf.clean_text('Reporte de Análisis y Optimización Financiera'), 0, 1, 'L')
        
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(212, 175, 55) 
        pdf.cell(0, 6, pdf.clean_text('FINANZAS - UNIVERSIDAD PRIVADA BOLIVIANA'), 0, 1, 'L')
        
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 5, pdf.clean_text(f'Fecha de Compilación: {date.today().strftime("%d/%m/%Y")} | Entorno Técnico Avanzado v2.0'), 0, 1, 'L')
        pdf.ln(6)
        
        if 'prices_full' not in st.session_state or st.session_state['prices_full'] is None:
            pdf.set_font('Arial', 'I', 11)
            pdf.set_text_color(150, 0, 0)
            pdf.cell(0, 10, pdf.clean_text('Falta de registros activos. Es requerido procesar los módulos de análisis en la aplicación.'), 0, 1, 'L')
            return pdf.output(dest='S').encode('latin-1')

        # Extracción de variables globales de la sesión activa
        selected_assets = st.session_state.get('selected_assets_m1', [])
        prices_full = st.session_state['prices_full']
        f_start = st.session_state.get('filt_start_m1', prices_full.index.min().date())
        f_end = st.session_state.get('filt_end_m1', prices_full.index.max().date())
        ann_factor_val = st.session_state.get('sel_freq', 'Diaria (252)')
        
        prices_selected = prices_full.loc[f_start:f_end][selected_assets]
        returns = prices_selected.pct_change().dropna()
        rf_rate_m1 = get_dynamic_rf(prices_selected)
        
        # Recálculo matemático interno para la consistencia total del Módulo 2
        mean_returns = returns.mean() * ann_factor
        cov_matrix = returns.cov() * ann_factor
        asset_vols = returns.std() * np.sqrt(ann_factor)
        num_activos = len(selected_assets)
        
        allow_shorts_val = st.session_state.get('chk_shorts_m2', False)
        max_weight_val = st.session_state.get('sld_max_weight_m2', 100.0)
        if max_weight_val > 1.0:
            max_weight_val = max_weight_val / 100.0
        lower_b = -0.5 if allow_shorts_val else 0.0
        
        bounds_pdf = [(lower_b, max_weight_val) for _ in range(num_activos)]
        cons_pdf = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        res_gmv_pdf = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))), num_activos * [1.0 / num_activos], method='SLSQP', bounds=bounds_pdf, constraints=cons_pdf)
        w_gmv_pdf = res_gmv_pdf.x
        ret_gmv_pdf, vol_gmv_pdf = np.sum(mean_returns * w_gmv_pdf), np.sqrt(np.dot(w_gmv_pdf.T, np.dot(cov_matrix, w_gmv_pdf)))
        sharpe_gmv_pdf = (ret_gmv_pdf - rf_rate_m1) / vol_gmv_pdf
        
        res_ms_pdf = minimize(lambda w: -((np.sum(mean_returns * w) - rf_rate_m1) / np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))), num_activos * [1.0 / num_activos], method='SLSQP', bounds=bounds_pdf, constraints=cons_pdf)
        w_ms_pdf = res_ms_pdf.x
        ret_ms_pdf, vol_ms_pdf = np.sum(mean_returns * w_ms_pdf), np.sqrt(np.dot(w_ms_pdf.T, np.dot(cov_matrix, w_ms_pdf)))
        sharpe_ms_pdf = (ret_ms_pdf - rf_rate_m1) / vol_ms_pdf
        
        w_ew_pdf = np.ones(num_activos) / num_activos
        ret_ew_pdf, vol_ew_pdf = np.sum(mean_returns * w_ew_pdf), np.sqrt(np.dot(w_ew_pdf.T, np.dot(cov_matrix, w_ew_pdf)))
        sharpe_ew_pdf = (ret_ew_pdf - rf_rate_m1) / vol_ew_pdf
        
        # Determinación de la estructura del Portafolio Completo
        calc_mode_val = st.session_state.get('rad_calc_m2', 'Por Coeficiente de Aversión al Riesgo (A)')
        A_coef_val = st.session_state.get('num_A_coef', 3.0)
        if calc_mode_val == "Por Coeficiente de Aversión al Riesgo (A)":
            y_opt_pdf = (ret_ms_pdf - rf_rate_m1) / (A_coef_val * (vol_ms_pdf ** 2))
            meta_allocation = f"Optima por Coeficiente de Aversion (A = {A_coef_val:.1f})"
        else:
            y_opt_pdf = st.session_state.get('sld_y_opt', 100.0) / 100.0
            meta_allocation = f"Manual parametrica (y = {y_opt_pdf * 100:.1f}%)"
            
        weight_rf_pdf = 1.0 - y_opt_pdf
        comp_weights_pdf = w_ms_pdf * y_opt_pdf

        # --------------------------------------------
        # RESUMEN DE PARÁMETROS GENERALES DE ENTORNO
        # --------------------------------------------
        pdf.agregar_subtitulo('Información General del Entorno de Análisis')
        pdf.set_font('Arial', '', 9.5)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 5, pdf.clean_text(f"- Indice de Referencia Seleccionado (Benchmark): {benchmark_ticker}"), 0, 1, 'L')
        pdf.cell(0, 5, pdf.clean_text(f"- Tasa Libre de Riesgo Aplicada (Risk-Free Rate): {rf_rate_m1 * 100:.2f}%"), 0, 1, 'L')
        pdf.cell(0, 5, pdf.clean_text(f"- Frecuencia Operativa de Datos y Anualización: {ann_factor_val}"), 0, 1, 'L')
        pdf.cell(0, 5, pdf.clean_text(f"- Metodologia de Asignacion de Cartera Completa: {meta_allocation}"), 0, 1, 'L')
        pdf.ln(5)

        # ============================================
        # SECCIÓN I: DIAGNÓSTICO DE RIESGO Y RENTABILIDAD
        # ============================================
        pdf.agregar_titulo_seccion('I. Diagnóstico Transversal de Riesgo y Rentabilidad')
        
        texto_intro_m1 = (
            f"El presente apartado resume el comportamiento cuantitativo del universo de activos seleccionado "
            f"en el período comprendido entre {f_start.strftime('%d/%m/%Y')} y {f_end.strftime('%d/%m/%Y')}. "
            f"Los indicadores estadísticos permiten caracterizar el perfil de distribución de los rendimientos."
        )
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, pdf.clean_text(texto_intro_m1))
        pdf.ln(4)

        pdf.agregar_subtitulo('Métricas de Rendimiento y Riesgo Anualizadas')
        
        pdf.set_fill_color(0, 32, 96) 
        pdf.set_text_color(255, 255, 255) 
        pdf.set_font('Arial', 'B', 9)
        
        col_w_asset = 25
        col_w_metric = 41
        
        pdf.cell(col_w_asset, 7, pdf.clean_text('Activo'), 1, 0, 'C', True)
        pdf.cell(col_w_metric, 7, pdf.clean_text('Retorno Arit. (%)'), 1, 0, 'C', True)
        pdf.cell(col_w_metric, 7, pdf.clean_text('Retorno Geo. (%)'), 1, 0, 'C', True)
        pdf.cell(col_w_metric, 7, pdf.clean_text('Volatilidad (%)'), 1, 0, 'C', True)
        pdf.cell(col_w_metric, 7, pdf.clean_text('Max Drawdown (%)'), 1, 1, 'C', True)
        
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(30, 30, 30)
        
        fill_toggle = False
        for asset in selected_assets:
            if fill_toggle:
                pdf.set_fill_color(245, 247, 250) 
            else:
                pdf.set_fill_color(255, 255, 255)
            
            ret_m1_arith = ((1 + returns[asset].mean()) ** ann_factor - 1) * 100
            ret_m1_geo = (((prices_selected[asset].iloc[-1] / prices_selected[asset].iloc[0]) ** (ann_factor / len(returns))) - 1) * 100
            vol_m1_ann = (returns[asset].std() * np.sqrt(ann_factor)) * 100
            cum_r = (1 + returns[asset]).cumprod()
            max_dd_val = (((cum_r - cum_r.cummax()) / cum_r.cummax()).min()) * 100
            
            pdf.cell(col_w_asset, 6, pdf.clean_text(asset), 1, 0, 'C', True)
            pdf.cell(col_w_metric, 6, pdf.clean_text(f"{ret_m1_arith:.2f}%"), 1, 0, 'C', True)
            pdf.cell(col_w_metric, 6, pdf.clean_text(f"{ret_m1_geo:.2f}%"), 1, 0, 'C', True)
            pdf.cell(col_w_metric, 6, pdf.clean_text(f"{vol_m1_ann:.2f}%"), 1, 0, 'C', True)
            pdf.cell(col_w_metric, 6, pdf.clean_text(f"{max_dd_val:.2f}%"), 1, 1, 'C', True)
            fill_toggle = not fill_toggle
            
        pdf.ln(6)

        # ============================================
        # SECCIÓN II: ASIGNACIÓN Y OPTIMIZACIÓN DE PESOS
        # ============================================
        pdf.agregar_titulo_seccion('II. Modelación y Optimización Moderna de Portafolios')
        
        texto_intro_m2 = (
            "Se implementaron algoritmos de optimización cuadrática basados en la Teoría Moderna de "
            "Portafolios (MPT) de Markowitz para deducir las asignaciones de capital eficientes bajo estructuras "
            "de diversificación paramétrica."
        )
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, pdf.clean_text(texto_intro_m2))
        pdf.ln(4)

        def calc_dr_pdf(weights, vols, p_vol):
            weighted_vol = np.sum(np.abs(weights) * vols)
            dr = weighted_vol / p_vol
            return dr, 1.0 - (1.0 / dr)
            
        dr_gmv, db_gmv = calc_dr_pdf(w_gmv_pdf, asset_vols, vol_gmv_pdf)
        dr_ms, db_ms = calc_dr_pdf(w_ms_pdf, asset_vols, vol_ms_pdf)
        dr_ew, db_ew = calc_dr_pdf(w_ew_pdf, asset_vols, vol_ew_pdf)

        # 1. ORDEN: Comparativa de Estrategias y Ratios de Diversificación
        pdf.agregar_subtitulo('Comparativa de Estrategias y Ratios de Diversificación')
        
        pdf.set_fill_color(0, 32, 96)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 9)
        
        col_w_m2_metric = 60
        col_w_m2_val = 40
        
        pdf.cell(col_w_m2_metric, 7, pdf.clean_text('Métrica de Frontera Eficiente'), 1, 0, 'L', True)
        pdf.cell(col_w_m2_val, 7, pdf.clean_text('Mínima Varianza'), 1, 0, 'C', True)
        pdf.cell(col_w_m2_val, 7, pdf.clean_text('Máximo Sharpe'), 1, 0, 'C', True)
        pdf.cell(col_w_m2_val, 7, pdf.clean_text('Equiponderado (1/N)'), 1, 1, 'C', True)
        
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(30, 30, 30)
        
        metricas_portafolio = [
            ('Retorno Esperado Anualizado (%)', ret_gmv_pdf * 100, ret_ms_pdf * 100, ret_ew_pdf * 100, '%'),
            ('Volatilidad Anualizada (%)', vol_gmv_pdf * 100, vol_ms_pdf * 100, vol_ew_pdf * 100, '%'),
            ('Ratio de Sharpe de Mercado', sharpe_gmv_pdf, sharpe_ms_pdf, sharpe_ew_pdf, 'num'),
            ('Ratio de Diversificación (DR)', dr_gmv, dr_ms, dr_ew, 'num'),
            ('Beneficio de Diversificación (%)', db_gmv * 100, db_ms * 100, db_ew * 100, '%')
        ]
        
        fill_toggle = False
        for label, v_gmv, v_ms, v_ew, m_type in metricas_portafolio:
            if fill_toggle:
                pdf.set_fill_color(245, 247, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
                
            if m_type == '%':
                f_gmv, f_ms, f_ew = f"{v_gmv:.2f}%", f"{v_ms:.2f}%", f"{v_ew:.2f}%"
            else:
                f_gmv, f_ms, f_ew = f"{v_gmv:.3f}", f"{v_ms:.3f}", f"{v_ew:.3f}"
                
            pdf.cell(col_w_m2_metric, 6, pdf.clean_text(label), 1, 0, 'L', True)
            pdf.cell(col_w_m2_val, 6, pdf.clean_text(f_gmv), 1, 0, 'C', True)
            pdf.cell(col_w_m2_val, 6, pdf.clean_text(f_ms), 1, 0, 'C', True)
            pdf.cell(col_w_m2_val, 6, pdf.clean_text(f_ew), 1, 1, 'C', True)
            fill_toggle = not fill_toggle
            
        pdf.ln(4)

        # 2. ORDEN: distribución transversal de capital por estrategia (Gráfico de barras)
        if pdf.get_y() > 195:
            pdf.add_page()
            
        try:
            fig_plt1, ax_plt1 = plt.subplots(figsize=(7, 3.1))
            x_idx = np.arange(len(selected_assets))
            b_width = 0.23
            ax_plt1.bar(x_idx - b_width, w_gmv_pdf * 100, b_width, label='Mínima Varianza', color='#002060')
            ax_plt1.bar(x_idx, w_ms_pdf * 100, b_width, label='Máximo Sharpe', color='#63b3ed')
            ax_plt1.bar(x_idx + b_width, w_ew_pdf * 100, b_width, label='Equiponderado', color='#e53e3e')
            ax_plt1.set_ylabel('Peso Asignado (%)', fontsize=8.5)
            ax_plt1.set_title('Distribución Transversal de Capital por Estrategia', fontsize=9.5, fontweight='bold', color='#002060')
            ax_plt1.set_xticks(x_idx)
            ax_plt1.set_xticklabels(selected_assets, fontsize=8, rotation=35)
            ax_plt1.legend(fontsize=7.5)
            ax_plt1.grid(axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_c1:
                plt.savefig(tmp_c1.name, dpi=300)
                c1_path = tmp_c1.name
            plt.close(fig_plt1)
            
            pdf.image(c1_path, x=15, y=pdf.get_y(), w=180)
            pdf.set_y(pdf.get_y() + 82)
            os.unlink(c1_path)
        except Exception as e:
            pdf.cell(0, 6, pdf.clean_text(f"[Gráfico de Ponderaciones Omitido por Error: {e}]"), 0, 1, 'L')
        
        pdf.ln(4)

        # 3 y 4. ORDEN: Composición de la Cartera Completa (Posiciones Activas) + Dona en paralelo
        if pdf.get_y() > 185:
            pdf.add_page()
            
        pdf.agregar_subtitulo('Composición de la Cartera Completa (Posiciones Activas)')
        
        start_y_comp = pdf.get_y()
        
        # Renderizado de la tabla de Cartera Completa (Lado Izquierdo)
        pdf.set_fill_color(0, 32, 96)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 9)
        pdf.set_x(15)
        pdf.cell(55, 7, pdf.clean_text('Activo / Instrumento'), 1, 0, 'L', True)
        pdf.cell(30, 7, pdf.clean_text('Peso (%)'), 1, 1, 'C', True)
        
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(30, 30, 30)
        fill_toggle_comp = False
        
        for idx, asset in enumerate(selected_assets):
            w_asset_comp = comp_weights_pdf[idx] * 100
            if abs(w_asset_comp) >= 0.01:
                if fill_toggle_comp:
                    pdf.set_fill_color(245, 247, 250)
                else:
                    pdf.set_fill_color(255, 255, 255)
                pdf.set_x(15)
                pdf.cell(55, 6, pdf.clean_text(asset), 1, 0, 'L', True)
                pdf.cell(30, 6, pdf.clean_text(f"{w_asset_comp:.2f}%"), 1, 1, 'C', True)
                fill_toggle_comp = not fill_toggle_comp
                
        w_rf_comp = weight_rf_pdf * 100
        if abs(w_rf_comp) >= 0.01:
            if fill_toggle_comp:
                pdf.set_fill_color(245, 247, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_x(15)
            pdf.cell(55, 6, pdf.clean_text('Activo Libre de Riesgo'), 1, 0, 'L', True)
            pdf.cell(30, 6, pdf.clean_text(f"{w_rf_comp:.2f}%"), 1, 1, 'C', True)
            
        end_y_table = pdf.get_y()
        
        # Renderizado del Gráfico de Dona de Cartera Completa (Lado Derecho)
        try:
            labels_pie = []
            sizes_pie = []
            for idx, asset in enumerate(selected_assets):
                w_val = comp_weights_pdf[idx] * 100
                if w_val >= 0.01:
                    labels_pie.append(asset)
                    sizes_pie.append(w_val)
            if weight_rf_pdf * 100 >= 0.01:
                labels_pie.append('Activo L.R.')
                sizes_pie.append(weight_rf_pdf * 100)
                
            fig_pie, ax_pie = plt.subplots(figsize=(3.2, 2.7))
            wedges, texts, autotexts = ax_pie.pie(
                sizes_pie, 
                labels=labels_pie, 
                autopct='%1.1f%%', 
                startangle=90,
                textprops=dict(color="black", size=7),
                wedgeprops=dict(width=0.4, edgecolor='white')
            )
            ax_pie.set_title('Distribución (Posiciones en Largo)', fontsize=8.5, fontweight='bold', color='#002060')
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_pie:
                plt.savefig(tmp_pie.name, dpi=300)
                pie_path = tmp_pie.name
            plt.close(fig_pie)
            
            pdf.image(pie_path, x=110, y=start_y_comp, w=85)
            end_y_chart = start_y_comp + 74
            os.unlink(pie_path)
        except Exception as e:
            end_y_chart = start_y_comp
            pdf.set_x(110)
            pdf.set_y(start_y_comp)
            pdf.cell(0, 6, pdf.clean_text(f"[Gráfico omitido por error: {e}]"), 0, 1, 'L')
            
        pdf.set_y(max(end_y_table, end_y_chart) + 6)

        # ============================================
        # SECCIÓN III: MODELO CAPM Y VALUACIÓN DE ACTIVOS
        # ============================================
        if pdf.get_y() > 210:
            pdf.add_page()
            
        pdf.agregar_titulo_seccion('III. Estructura de Equilibrio CAPM y Análisis de Alfas')
        
        texto_intro_m3 = (
            "Se implementaron regresiones lineales para contrastar las tasas históricas frente a los "
            "requerimientos de rentabilidad del Capital Asset Pricing Model, identificando distorsiones "
            "de valuación mediante el Alfa de Jensen."
        )
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, pdf.clean_text(texto_intro_m3))
        pdf.ln(4)
        
        pdf.agregar_subtitulo('Matriz de Sensibilidad Sistemática y Diagnóstico de Mercado')
        
        pdf.set_fill_color(0, 32, 96)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 8.5)
        
        col_w_m3_asset = 20
        col_w_m3_data = 28
        col_w_m3_txt = 58
        
        pdf.cell(col_w_m3_asset, 7, pdf.clean_text('Activo'), 1, 0, 'C', True)
        pdf.cell(col_w_m3_data, 7, pdf.clean_text('Beta (β)'), 1, 0, 'C', True)
        pdf.cell(col_w_m3_data, 7, pdf.clean_text('Rnd. CAPM'), 1, 0, 'C', True)
        pdf.cell(col_w_m3_data, 7, pdf.clean_text('Alfa Jensen'), 1, 0, 'C', True)
        pdf.cell(col_w_m3_data, 7, pdf.clean_text('VaR Hist 95%'), 1, 0, 'C', True)
        pdf.cell(col_w_m3_txt, 7, pdf.clean_text('Diagnóstico de Valuación'), 1, 1, 'C', True)
        
        pdf.set_font('Arial', '', 8.5)
        pdf.set_text_color(30, 30, 30)
        
        betas_list = []
        
        bench_data_m3 = st.session_state.get('benchmark_data')
        if bench_data_m3 is not None and not bench_data_m3.empty:
            returns_b_m3 = bench_data_m3.loc[f_start:f_end].pct_change().dropna().iloc[:, 0]
            common_idx_m3 = returns.index.intersection(returns_b_m3.index)
            
            rb_aligned = returns_b_m3.loc[common_idx_m3]
            m_return_ann = (1 + rb_aligned.mean()) ** ann_factor - 1
            m_risk_prem = m_return_ann - rf_rate_m1
            
            fill_toggle = False
            for asset in selected_assets:
                if fill_toggle:
                    pdf.set_fill_color(245, 247, 250)
                else:
                    pdf.set_fill_color(255, 255, 255)
                
                ra_aligned = returns[asset].loc[common_idx_m3]
                slope_b, _, _, _, _ = linregress(rb_aligned, ra_aligned)
                betas_list.append(slope_b)
                
                r_hist_m3 = ((1 + ra_aligned.mean()) ** ann_factor - 1)
                r_capm_m3 = rf_rate_m1 + slope_b * m_risk_prem
                alpha_j_m3 = r_hist_m3 - r_capm_m3
                var_95_val = ra_aligned.quantile(0.05) * 100
                
                status_txt = "Subvalorado (Sobre SML)" if alpha_j_m3 > 0 else "Sobrevalorado (Bajo SML)"
                
                pdf.cell(col_w_m3_asset, 6, pdf.clean_text(asset), 1, 0, 'C', True)
                pdf.cell(col_w_m3_data, 6, pdf.clean_text(f"{slope_b:.3f}"), 1, 0, 'C', True)
                pdf.cell(col_w_m3_data, 6, pdf.clean_text(f"{r_capm_m3*100:.2f}%"), 1, 0, 'C', True)
                pdf.cell(col_w_m3_data, 6, pdf.clean_text(f"{alpha_j_m3*100:.2f}%"), 1, 0, 'C', True)
                pdf.cell(col_w_m3_data, 6, pdf.clean_text(f"{var_95_val:.2f}%"), 1, 0, 'C', True)
                pdf.cell(col_w_m3_txt, 6, pdf.clean_text(status_txt), 1, 1, 'L', True)
                fill_toggle = not fill_toggle

        pdf.ln(4)

        # --------------------------------------------
        # ADICIÓN DE GRÁFICO 2: SECURITY MARKET LINE
        # --------------------------------------------
        if pdf.get_y() > 190:
            pdf.add_page()
            
        try:
            fig_plt2, ax_plt2 = plt.subplots(figsize=(7, 3.2))
            min_b = min(betas_list + [0.0, 1.0]) - 0.15
            max_b = max(betas_list + [1.0]) + 0.15
            b_space = np.linspace(min_b, max_b, 15)
            sml_space = (rf_rate_m1 + b_space * m_risk_prem) * 100
            
            ax_plt2.plot(b_space, sml_space, color='#002060', linewidth=1.8, label='Security Market Line (SML)')
            ax_plt2.scatter(1.0, m_return_ann * 100, color='#D4AF37', marker='*', s=130, label='Mercado', zorder=5)
            
            for idx, asset in enumerate(selected_assets):
                b_val = betas_list[idx]
                r_val_hist = (((1 + returns[asset].loc[common_idx_m3].mean()) ** ann_factor - 1)) * 100
                r_val_capm = (rf_rate_m1 + b_val * m_risk_prem) * 100
                pt_color = 'green' if (r_val_hist > r_val_capm) else 'red'
                ax_plt2.scatter(b_val, r_val_hist, color=pt_color, edgecolors='black', s=45, zorder=4)
                ax_plt2.text(b_val, r_val_hist + 0.4, asset, fontsize=7, ha='center')
                
            ax_plt2.set_xlabel('Riesgo Sistemático (Beta β)', fontsize=8.5)
            ax_plt2.set_ylabel('Retorno Esperado Anualizado (%)', fontsize=8.5)
            ax_plt2.set_title('Posicionamiento del Universo de Activos frente a la SML', fontsize=9.5, fontweight='bold', color='#002060')
            ax_plt2.grid(linestyle='--', alpha=0.4)
            ax_plt2.legend(fontsize=7.5)
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_c2:
                plt.savefig(tmp_c2.name, dpi=300)
                c2_path = tmp_c2.name
            plt.close(fig_plt2)
            
            pdf.image(c2_path, x=15, y=pdf.get_y(), w=180)
            pdf.set_y(pdf.get_y() + 85)
            os.unlink(c2_path)
        except Exception as e:
            pdf.cell(0, 6, pdf.clean_text(f"[Gráfico de la Recta SML Omitido por Error: {e}]"), 0, 1, 'L')

        pdf.ln(8)
        
        # Firma Institucional de Cierre
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(0, 32, 96)
        pdf.cell(0, 4, pdf.clean_text('UNIVERSIDAD PRIVADA BOLIVIANA - LA PAZ'), 0, 1, 'R')
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 4, pdf.clean_text('Documento compilado y validado mediante algoritmos de optimización matricial.'), 0, 1, 'R')
        
        return pdf.output(dest='S').encode('latin-1')


    # --------------------------------------------
    # RENDERIZADO INTERFAZ DE USUARIO (UI)
    # --------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("Reportes y Cierre Corporativo")
    
    if st.sidebar.button("Compilar Resumen Ejecutivo (PDF)", key="btn_compile_pdf_global"):
        try:
            with st.spinner("Compilando matrices y formateando reporte en PDF..."):
                pdf_data_bytes = generar_pdf_ejecutivo()
                st.session_state['reporte_pdf_ready'] = pdf_data_bytes
                st.sidebar.success("Estructura de Resumen Ejecutivo compilada con éxito.")
        except Exception as e:
            st.sidebar.error(f"Falla en la generación del PDF corporativo: {e}")

    if st.session_state.get('reporte_pdf_ready') is not None:
        st.sidebar.download_button(
            label="Descargar Resumen Ejecutivo (PDF)",
            data=st.session_state['reporte_pdf_ready'],
            file_name=f"Resumen_Ejecutivo_Inversion_{date.today().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            key="dl_btn_pdf_sidebar"
        )

# Disclaimer Institucional en la interfaz
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.caption(
        "⚠️ **Aviso de Responsabilidad:** Esta herramienta informática fue desarrollada con fines "
        "estrictamente académicos para la materia de Finanzas de la Universidad Privada Boliviana - La Paz. "
        "Las optimizaciones de Markowitz, simulaciones estocásticas de Monte Carlo y proyecciones del modelo CAPM "
        "se sustentan en datos históricos y asunciones matemáticas. No constituyen asesoramiento "
        "financiero profesional ni recomendaciones de inversión."
    )
# ============================================
# PIE DE PÁGINA
# ============================================
st.divider()
st.caption("Trabajo Final | Finanzas I | UPB La Paz - 2026")
