# =============================================================================
# COMANDOS PARA ENVIAR PROJETO PARA GITHUB
# =============================================================================
# 
# Execute estes comandos no terminal/PowerShell na pasta do projeto
# Certifique-se de estar na pasta: AI-SRAG_indicium_project
#
# =============================================================================

# 1. NAVEGUE ATÉ A PASTA DO PROJETO
cd "c:\Users\root\Desktop\Engenharia de dados - indicium\AI-SRAG_indicium_project"

# 2. INICIALIZAR REPOSITÓRIO GIT (se ainda não foi feito)
git init

# 3. CONFIGURAR REMOTE PARA SEU REPOSITÓRIO GITHUB
git remote add origin https://github.com/Nico-Draagron/AI-SRAG_indicium_project.git

# 4. VERIFICAR SE O .env ESTÁ SENDO IGNORADO (IMPORTANTE!)
# Este comando deve mostrar que .env está listado (protegido)
git status

# 5. ADICIONAR TODOS OS ARQUIVOS AO STAGING
git add .

# 6. FAZER COMMIT COM MENSAGEM DESCRITIVA
git commit -m "feat: Implementação completa do sistema AI-SRAG para certificação Indicium

- Sistema híbrido SQL + RAG com OpenAI GPT-4o-mini
- Pipeline completo Bronze → Silver → Gold
- Vector Store com Databricks Vector Search
- Orquestrador LangGraph com roteamento inteligente
- Auditoria completa e tratamento de exceções
- RAG totalmente opcional e configurável
- Guardrails de segurança em consultas SQL
- Sistema de relatórios automatizados
- Estrutura modular e escalável"

# 7. ENVIAR PARA O GITHUB (BRANCH MAIN)
git push -u origin main

# =============================================================================
# COMANDOS ALTERNATIVOS (se houver problemas)
# =============================================================================

# Se o repositório GitHub já existir com conteúdo:
git pull origin main --allow-unrelated-histories
git push origin main

# Para forçar push (CUIDADO - só se necessário):
git push origin main --force

# Para verificar status e logs:
git status
git log --oneline

# =============================================================================
# VERIFICAÇÕES IMPORTANTES ANTES DO PUSH
# =============================================================================

# 1. Verificar se .env está sendo ignorado:
git status | findstr ".env"
# (NÃO deve aparecer .env na lista, apenas .env.example)

# 2. Verificar se arquivos sensíveis não estão sendo commitados:
git ls-files | findstr -E "\.(env|key|pem|cert|log)$"
# (Não deve retornar nada ou apenas .env.example)

# 3. Testar se .gitignore está funcionando:
echo "test" > .env
git status
# (.env NÃO deve aparecer na lista de arquivos para commit)
del .env

# =============================================================================
# APÓS O PUSH - CONFIGURAR NO GITHUB
# =============================================================================

# 1. No repositório GitHub, vá em: Settings > Secrets and variables > Actions
# 2. Adicione as seguintes secrets:
#    - OPENAI_API_KEY: sua chave OpenAI
#    - DATABRICKS_TOKEN: seu token Databricks (se necessário)
#    - TAVILY_API_KEY: sua chave Tavily (se usar web search)

# =============================================================================