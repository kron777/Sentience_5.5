#!/usr/bin/env bash
# ii.sh – “init & ingest” – turn the current folder into a fresh GitHub repo
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------- user-adjustable defaults (change here if you want) -------------
GITHUB_USER=""               # leave empty to be asked
REPO_NAME=""                 # leave empty = folder name
PRIVATE="false"              # "true"  → private repo
                             # "false" → public repo
REMOTE_NAME="origin"         # the usual
# ---------------------------------------------------------------------------

# pretty colours
R=$(tput setaf 1); G=$(tput setaf 2); Y=$(tput setaf 3); B=$(tput setaf 4); N=$(tput sgr0)

# helper: ask a question and return the reply
ask(){ read -rp "$1 [$2]: " v; echo "${v:-$2}"; }

# 1. discover repo name ------------------------------------------------------
DIRNAME=$(basename "$PWD")
REPO_NAME=${REPO_NAME:-$(ask "Repository name" "$DIRNAME")}

# 2. GitHub username ---------------------------------------------------------
if [ -z "$GITHUB_USER" ]; then
   # try to grab it from global config or ask
   GITHUB_USER=$(git config --global user.name 2>/dev/null || true)
   [ -z "$GITHUB_USER" ] && GITHUB_USER=$(ask "GitHub username" "")
fi

# 3. private / public --------------------------------------------------------
PRIVATE=$(ask "Private repo" "$PRIVATE")

# 4. create the remote repo via GitHub CLI (if installed) --------------------
if command -v gh &>/dev/null; then
   echo -e "\n${G}Creating GitHub repo ${B}$GITHUB_USER/$REPO_NAME${N}"
   gh repo create "$REPO_NAME" ${PRIVATE:+-p} -d "Autonomous sentience framework" -y
else
   echo -e "\n${Y}GitHub CLI (gh) not found – create the repo manually on the web:${N}"
   echo "   https://github.com/new   name: $REPO_NAME   private: $PRIVATE"
   echo "   Then press ENTER to continue ..."; read -r
fi

# 5. local git setup ---------------------------------------------------------
if [ ! -d .git ]; then
   echo -e "\n${G}Initialising local Git repository${N}"
   git init
fi

# 6. unified .gitignore ------------------------------------------------------
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# ROS
build/
install/
log/
.ros/
*.bag
*.pcd

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
EOF

# 7. first commit ------------------------------------------------------------
echo -e "\n${G}Adding everything and committing${N}"
git add .
git commit -m "Initial commit – autonomous sentience framework v5.5"

# 8. connect local → remote --------------------------------------------------
REMOTE_URL="git@github.com:$GITHUB_USER/$REPO_NAME.git"
git remote get-url "$REMOTE_NAME" &>/dev/null && git remote remove "$REMOTE_NAME"
git remote add "$REMOTE_NAME" "$REMOTE_URL"

# 9. push --------------------------------------------------------------------
echo -e "\n${G}Pushing to GitHub${N}"
git branch -M main
git push -u "$REMOTE_NAME" main

echo -e "\n${G}Done!${N}  Your code is live at https://github.com/$GITHUB_USER/$REPO_NAME"
