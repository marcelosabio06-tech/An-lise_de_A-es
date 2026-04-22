# -*- coding: utf-8 -*-
"""
📊 Analisador de Ações B3 - Curto Prazo (~7 dias)
Desenvolvido para Streamlit | Coleta gratuita via yfinance
Autor: Assistente de Engenharia de Software
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from io import BytesIO
import warnings
import sys

# Suprime warnings que poluem o log
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Garante encoding UTF-8 para evitar erros de caracteres
if sys.stdout.encoding != 'UTF-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="📈 Analisador B3 - Curto Prazo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FUNÇÕES MODULARES
# =============================================================================

def fetch_data(ticker: str, period: str = "120d"):
    """
    Coleta dados históricos de preço e volume via yfinance.
    Retorna DataFrame com OHLCV e dict de info, ou (None, None) em caso de erro.
    """
    try:
        # Adiciona suffixo .SA para tickers da B3 se necessário
        ticker_sa = f"{ticker}.SA" if not ticker.endswith(".SA") else ticker
        
        stock = yf.Ticker(ticker_sa)
        
        # Tenta buscar histórico com retry simples
        df = None
        for attempt in range(3):
            try:
                df = stock.history(period=period, interval="1d")
                if df is not None and not df.empty:
                    break
            except:
                if attempt == 2:
                    return None, None
                import time
                time.sleep(1)
        
        if df is None or df.empty:
            return None, None
        
        # Extrai info fundamental com try/except para evitar crash
        try:
            info = stock.info
        except:
            info = {}  # Fallback vazio
        
        return df, info
    except Exception as e:
        st.warning(f"⚠️ Dados parciais para {ticker}: {str(e)[:100]}")
        return None, None


def calc_technicals(df: pd.DataFrame) -> dict:
    """
    Calcula indicadores técnicos: RSI, MACD, EMAs, Bandas de Bollinger.
    Retorna dicionário com valores atuais e sinais.
    """
    if df is None or len(df) < 30:
        return {}
    
    df = df.copy()
    close = df['Close']
    volume = df['Volume']
    
    try:
        # RSI(14)
        rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]
    except:
        rsi = None
    
    try:
        # MACD (12, 26, 9)
        macd_ind = MACD(close=close)
        macd_line = macd_ind.macd().iloc[-1]
        macd_signal_vals = macd_ind.macd_signal()
        macd_signal = macd_signal_vals.iloc[-2] if len(macd_signal_vals) >= 2 else macd_line
        macd_histogram = macd_ind.macd_diff().iloc[-1]
    except:
        macd_line = macd_signal = macd_histogram = None
    
    try:
        # EMAs: 9, 21, 200
        ema9 = EMAIndicator(close=close, window=9).ema_indicator().iloc[-1]
        ema21 = EMAIndicator(close=close, window=21).ema_indicator().iloc[-1]
        ema200 = EMAIndicator(close=close, window=200).ema_indicator().iloc[-1] if len(close) >= 200 else close.iloc[-1]
    except:
        ema9 = ema21 = ema200 = None
    
    try:
        # Bandas de Bollinger (20, 2)
        bb = BollingerBands(close=close, window=20, window_dev=2)
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_upper = bb.bollinger_hband().iloc[-1]
    except:
        bb_lower = bb_upper = None
    
    # Volume: média móvel de 20 dias
    try:
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
    except:
        vol_ma20 = None
    current_vol = volume.iloc[-1] if len(volume) > 0 else None
    
    current_price = close.iloc[-1] if len(close) > 0 else None
    
    return {
        'preco_atual': float(current_price) if current_price is not None else None,
        'rsi_14': float(rsi) if rsi is not None else None,
        'macd_line': float(macd_line) if macd_line is not None else None,
        'macd_signal': float(macd_signal) if macd_signal is not None else None,
        'macd_histogram': float(macd_histogram) if macd_histogram is not None else None,
        'ema9': float(ema9) if ema9 is not None else None,
        'ema21': float(ema21) if ema21 is not None else None,
        'ema200': float(ema200) if ema200 is not None else None,
        'bb_lower': float(bb_lower) if bb_lower is not None else None,
        'bb_upper': float(bb_upper) if bb_upper is not None else None,
        'vol_atual': float(current_vol) if current_vol is not None else None,
        'vol_ma20': float(vol_ma20) if vol_ma20 is not None else None
    }


def check_fundamentals(info: dict) -> dict:
    """
    Avalia indicadores fundamentalistas extraídos do yfinance.
    Retorna dicionário com métricas e status de qualificação.
    """
    if not info or not isinstance(info, dict):
        return {'qualificada': False, 'razoes': ['Dados fundamentais indisponíveis']}
    
    # Extrai campos com fallback para None
    pe = info.get('trailingPE') or info.get('forwardPE')
    pb = info.get('priceToBook')
    roe = info.get('returnOnEquity')
    dy = info.get('dividendYield')
    debt_equity = info.get('debtToEquity')  # Proxy quando EBITDA não disponível
    net_margin = info.get('profitMargins')
    
    # Converte para valores utilizáveis
    roe_pct = (roe or 0) * 100 if isinstance(roe, (int, float)) else None
    dy_pct = (dy or 0) * 100 if isinstance(dy, (int, float)) else None
    margin_pct = (net_margin or 0) * 100 if isinstance(net_margin, (int, float)) else None
    
    # Critérios de qualificação fundamental
    criterios = []
    
    # P/L entre 5 e 20
    if isinstance(pe, (int, float)) and 5 <= pe <= 20:
        criterios.append(True)
    elif pe is None:
        criterios.append(None)  # Neutro se indisponível
    else:
        criterios.append(False)
    
    # ROE > 12%
    if isinstance(roe_pct, (int, float)) and roe_pct > 12:
        criterios.append(True)
    elif roe_pct is None:
        criterios.append(None)
    else:
        criterios.append(False)
    
    # Dívida/Equity < 100% (proxy para Dívida/EBITDA < 3.0x)
    if isinstance(debt_equity, (int, float)) and debt_equity < 100:
        criterios.append(True)
    elif debt_equity is None:
        criterios.append(None)
    else:
        criterios.append(False)
    
    # DY > 4% (opcional)
    if isinstance(dy_pct, (int, float)) and dy_pct > 4:
        criterios.append(True)
    elif dy_pct is None:
        criterios.append(None)
    else:
        criterios.append(False)
    
    # Conta indicadores dentro da faixa (ignora None)
    validos = [c for c in criterios if c is not None]
    dentro_faixa = sum(1 for c in validos if c is True)
    
    qualificada = dentro_faixa >= 3 if len(validos) >= 3 else (dentro_faixa >= 2 if validos else False)
    
    return {
        'pe': pe,
        'pb': pb,
        'roe_pct': roe_pct,
        'dy_pct': dy_pct,
        'debt_proxy': debt_equity,
        'margin_pct': margin_pct,
        'qualificada': qualificada,
        'razoes': [
            f"P/L={pe:.1f}" if isinstance(pe, (int, float)) else "P/L=N/A",
            f"ROE={roe_pct:.1f}%" if isinstance(roe_pct, (int, float)) else "ROE=N/A",
            f"Dívida/Proxy={debt_equity:.1f}x" if isinstance(debt_equity, (int, float)) else "Dívida=N/A",
            f"DY={dy_pct:.1f}%" if isinstance(dy_pct, (int, float)) else "DY=N/A"
        ]
    }


def generate_signal(tech: dict, fund: dict) -> tuple:
    """
    Gera sinal de compra baseado em ponderação 80% técnico / 20% fundamental.
    Retorna: (classificação, score_técnico, justificativa)
    """
    if not tech or not fund:
        return "❌ EVITAR", 0, "Dados insuficientes para análise"
    
    score = 0
    justificativas = []
    
    # 🔹 ANÁLISE TÉCNICA (Peso 80% - 5 critérios)
    
    # 1. RSI < 35 → sobrevenda (COMPRA)
    rsi = tech.get('rsi_14')
    if isinstance(rsi, (int, float)):
        if rsi < 35:
            score += 1
            justificativas.append(f"RSI={rsi:.1f} (sobrevenda)")
        elif rsi > 65:
            justificativas.append(f"RSI={rsi:.1f} (sobrecompra ⚠️)")
    
    # 2. MACD: cruzamento de alta + histograma positivo
    macd_l = tech.get('macd_line')
    macd_s = tech.get('macd_signal')
    macd_h = tech.get('macd_histogram')
    if all(isinstance(x, (int, float)) for x in [macd_l, macd_s, macd_h]):
        if macd_l > macd_s and macd_h > 0:
            score += 1
            justificativas.append("MACD cruzou para cima ✓")
    
    # 3. Preço > EMA9 > EMA21 → tendência de alta
    price = tech.get('preco_atual')
    ema9 = tech.get('ema9')
    ema21 = tech.get('ema21')
    if all(isinstance(x, (int, float)) for x in [price, ema9, ema21]):
        if price > ema9 > ema21:
            score += 1
            justificativas.append("Preço acima de EMA9 e EMA21 ✓")
    
    # 4. Bollinger: preço próximo da banda inferior
    bb_lower = tech.get('bb_lower')
    if all(isinstance(x, (int, float)) for x in [price, bb_lower]):
        if price <= bb_lower * 1.02:
            score += 1
            justificativas.append("Preço na banda inferior de Bollinger ✓")
    
    # 5. Volume > 1.2x média de 20 dias
    vol_atual = tech.get('vol_atual')
    vol_ma20 = tech.get('vol_ma20')
    if all(isinstance(x, (int, float)) for x in [vol_atual, vol_ma20]) and vol_ma20 > 0:
        if vol_atual > vol_ma20 * 1.2:
            score += 1
            justificativas.append(f"Volume {vol_atual/vol_ma20:.1f}x acima da média ✓")
    
    # 🔸 ANÁLISE FUNDAMENTAL (Peso 20% - filtro de qualidade)
    fund_razoes = fund.get('razoes', [])
    
    if not fund.get('qualificada', False):
        justificativas.append("Fundamental: atenção aos indicadores")
        # Reduz score se fundamental fraco
        if score >= 3:
            score -= 1
    else:
        justificativas.append("Fundamentalmente qualificada ✓")
    
    # 🎯 CLASSIFICAÇÃO FINAL
    if score >= 4 and fund.get('qualificada', False):
        classificacao = "✅ COMPRA CURTO PRAZO"
    elif score >= 2 and (rsi is None or rsi <= 65):
        classificacao = "⚠️ AGUARDAR"
    else:
        classificacao = "❌ EVITAR"
    
    # Monta justificativa final
    justificativa_final = "; ".join(justificativas[:5])  # Limita a 5 pontos
    justificativa_final += f" | Score Técnico: {score}/5"
    
    return classificacao, score, justificativa_final


def export_to_excel(results: list, raw_ dict) -> BytesIO:
    """
    Gera arquivo Excel com 3 abas: Resumo, Dados Brutos, Glossário.
    Retorna objeto BytesIO para download no Streamlit.
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Aba 1: Resumo
        df_resumo = pd.DataFrame(results)
        df_resumo.to_excel(writer, sheet_name='Resumo', index=False)
        
        # Aba 2: Dados Brutos
        if raw_
            df_brutos = pd.DataFrame(raw_data)
            df_brutos.to_excel(writer, sheet_name='Dados_Brutos', index=False)
        
        # Aba 3: Glossário
        glossario = pd.DataFrame({
            'Indicador': [
                'RSI(14)', 'MACD', 'EMA9/21/200', 'Bandas de Bollinger', 
                'P/L', 'ROE', 'Dívida/EBITDA', 'DY', 'Classificação'
            ],
            'Interpretação': [
                '<35: sobrevenda (compra); >65: sobrecompra (venda)',
                'Cruzamento MACD > Sinal + histograma positivo = confirmação de alta',
                'Preço > EMA9 > EMA21 = tendência de alta confirmada',
                'Preço na banda inferior = possível oportunidade de reversão',
                '5-20: faixa razoável; <5: barato; >25: caro',
                '>12%: boa eficiência; >15%: excelente',
                '<3.0x: saudável; >3.5x: risco elevado',
                '>4%: atrativo para renda; >6%: muito atrativo',
                'COMPRA: score técnico ≥4 + fundamental qualificado'
            ]
        })
        glossario.to_excel(writer, sheet_name='Glossario', index=False)
    
    output.seek(0)
    return output


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def main():
    # Cabeçalho
    st.title("📊 Analisador de Ações B3 - Curto Prazo (~7 dias)")
    st.markdown("""
    > **Objetivo**: Analisar 5 ações da B3 e identificar oportunidades de compra 
    > para operação de curto prazo, combinando indicadores técnicos (80%) e 
    > fundamentalistas (20% como filtro de qualidade).
    """)
    
    # Disclaimer obrigatório
    with st.expander("⚠️ LEIA ANTES DE USAR - DISCLAIMER IMPORTANTE", expanded=True):
        st.warning("""
        🔹 **Este aplicativo NÃO garante lucro**. O mercado de ações envolve riscos significativos.
        🔹 Indicadores são ferramentas de apoio à decisão, não sinais infalíveis.
        🔹 Operações de curto prazo (~7 dias) exigem monitoramento constante e gestão de risco.
        🔹 Defina sempre stop loss (2-3%) e tamanho de posição adequado ao seu capital.
        🔹 Consulte um advisor financeiro certificado antes de investir.
        🔹 Dados coletados via yfinance podem ter atraso ou indisponibilidade.
        """)
    
    # Sidebar: Entrada de tickers
    st.sidebar.header("🔍 Configuração")
    
    default_tickers = ["PETR4", "VALE3", "ITUB4", "BBDC4", "WEGE3"]
    tickers = []
    
    for i in range(5):
        ticker = st.sidebar.text_input(
            f"Ação {i+1}", 
            value=default_tickers[i] if i < len(default_tickers) else "",
            placeholder="Ex: PETR4",
            key=f"ticker_{i}"
        )
        if ticker.strip():
            tickers.append(ticker.strip().upper())
    
    analyze_btn = st.sidebar.button("🚀 ANALISAR AÇÕES", type="primary", use_container_width=True)
    
    # Processamento
    if analyze_btn and tickers:
        results = []
        raw_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(tickers):
            status_text.text(f"🔄 Analisando {ticker}... ({idx+1}/{len(tickers)})")
            
            # Coleta dados
            df, info = fetch_data(ticker)
            
            if df is None:
                results.append({
                    'Ticker': ticker,
                    'Preço Atual': 'N/A',
                    'Sinal Técnico': 'Dados indisponíveis',
                    'Score Técnico': 0,
                    'Filtro Fundamental': 'N/A',
                    'Classificação': '❌ EVITAR',
                    'Justificativa': 'Não foi possível coletar dados. Verifique o ticker.'
                })
                progress_bar.progress((idx + 1) / len(tickers))
                continue
            
            # Calcula indicadores
            tech = calc_technicals(df)
            fund = check_fundamentals(info)
            classificacao, score, justificativa = generate_signal(tech, fund)
            
            # Prepara resultado
            preco_fmt = f"R$ {tech.get('preco_atual'):.2f}" if isinstance(tech.get('preco_atual'), (int, float)) else "N/A"
            
            resultado = {
                'Ticker': ticker,
                'Preço Atual': preco_fmt,
                'Sinal Técnico': f"{score}/5",
                'Score Técnico': score,
                'Filtro Fundamental': '✓ Qualificada' if fund.get('qualificada') else '⚠️ Atenção',
                'Classificação': classificacao,
                'Justificativa': justificativa
            }
            results.append(resultado)
            
            # Armazena dados brutos para Excel
            raw_entry = {
                'Ticker': ticker,
                'Preço': tech.get('preco_atual'),
                'RSI_14': tech.get('rsi_14'),
                'MACD_Line': tech.get('macd_line'),
                'MACD_Signal': tech.get('macd_signal'),
                'EMA9': tech.get('ema9'),
                'EMA21': tech.get('ema21'),
                'EMA200': tech.get('ema200'),
                'BB_Lower': tech.get('bb_lower'),
                'Volume_Atual': tech.get('vol_atual'),
                'Volume_MA20': tech.get('vol_ma20'),
                'P/L': fund.get('pe'),
                'P/VP': fund.get('pb'),
                'ROE_%': fund.get('roe_pct'),
                'DY_%': fund.get('dy_pct'),
                'Dívida_Proxy': fund.get('debt_proxy')
            }
            raw_data.append(raw_entry)
            
            progress_bar.progress((idx + 1) / len(tickers))
        
        status_text.empty()
        progress_bar.empty()
        
        # Exibe resultados
        st.subheader("📋 Resultados da Análise")
        
        # Tabela interativa
        df_results = pd.DataFrame(results)
        st.dataframe(
            df_results[['Ticker', 'Preço Atual', 'Classificação', 'Score Técnico', 'Justificativa']],
            use_container_width=True,
            hide_index=True
        )
        
        # Resumo visual por classificação
        col1, col2, col3 = st.columns(3)
        compras = len([r for r in results if 'COMPRA' in r['Classificação']])
        aguardar = len([r for r in results if 'AGUARDAR' in r['Classificação']])
        evitar = len([r for r in results if 'EVITAR' in r['Classificação']])
        
        col1.metric("✅ Compra Curto Prazo", compras)
        col2.metric("⚠️ Aguardar", aguardar)
        col3.metric("❌ Evitar", evitar)
        
        # Botão de exportação Excel
        excel_file = export_to_excel(results, raw_data)
        
        st.download_button(
            label="📥 BAIXAR RELATÓRIO COMPLETO (Excel)",
            data=excel_file,
            file_name=f"analise_b3_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # Detalhes técnicos expandíveis
        with st.expander("🔍 Ver detalhes técnicos de cada ação"):
            for res in results:
                with st.container():
                    st.markdown(f"**{res['Ticker']}** - {res['Classificação']}")
                    st.code(res['Justificativa'], language=None)
                    st.divider()
    
    elif analyze_btn and not tickers:
        st.error("⚠️ Por favor, informe pelo menos 1 ticker para análise.")
    
    # Rodapé informativo
    st.markdown("---")
    st.caption("""
    📌 **Como interpretar**: 
    • **COMPRA CURTO PRAZO**: Sinal técnico forte (≥4/5) + fundamental qualificado. 
    • **AGUARDAR**: Sinal misto ou falta de confirmação de volume. 
    • **EVITAR**: RSI sobrecomprado, tendência de baixa ou risco fundamental alto.
    
    🔗 Fontes de dados: yfinance (Yahoo Finance) | Indicadores calculados com biblioteca `ta`
    """)


if __name__ == "__main__":
    main()
