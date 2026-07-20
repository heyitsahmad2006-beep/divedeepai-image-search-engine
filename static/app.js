const root = document.documentElement;
const savedTheme = localStorage.getItem('divedeepai-theme');
if (savedTheme) root.dataset.theme = savedTheme;
else if (matchMedia('(prefers-color-scheme: dark)').matches) root.dataset.theme = 'dark';

document.querySelector('#themeToggle').addEventListener('click', () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('divedeepai-theme', root.dataset.theme);
});

async function checkHealth() {
  try {
    const health = await fetch('/health').then(r => r.json());
    const badge = document.querySelector('#indexBadge');
    badge.classList.toggle('ready', health.index_ready);
    badge.innerHTML = `<i></i>${health.index_ready ? 'Index ready' : 'Index needed'}`;
  } catch (_) { document.querySelector('#indexBadge').innerHTML = '<i></i> Offline'; }
}
checkHealth();

document.querySelectorAll('.suggestions button').forEach(button => button.addEventListener('click', () => {
  document.querySelector('#query').value = button.textContent;
  document.querySelector('#search').requestSubmit();
}));

document.querySelector('#search').addEventListener('submit', async event => {
  event.preventDefault();
  const query = document.querySelector('#query').value.trim();
  const status = document.querySelector('#status');
  const results = document.querySelector('#results');
  const title = document.querySelector('#resultsTitle');
  const latency = document.querySelector('#latency');
  title.textContent = `Results for “${query}”`;
  status.textContent = 'Finding the closest visual matches…'; latency.textContent = '';
  results.innerHTML = '<div class="loader"></div>';
  try {
    const response = await fetch('/search', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,top_k:Number(document.querySelector('#topK').value)})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Search failed.');
    status.textContent = `${data.results.length} semantic matches`;
    latency.textContent = `${data.latency_ms} ms`;
    results.innerHTML = '';
    data.results.forEach((item,index) => {
      const percent = Math.max(0,Math.min(100,item.similarity_score * 100));
      results.insertAdjacentHTML('beforeend', `<article class="card" style="animation-delay:${Math.min(index*35,350)}ms"><div class="image-wrap"><img src="${item.image_url}" alt="${item.image_name}" loading="lazy"><span class="rank">#${item.rank}</span></div><div class="card-body"><div class="card-title">${item.image_name}</div><div class="score-row"><span>${item.similarity_score.toFixed(3)}</span><div class="score-track"><div class="score-fill" style="width:${percent}%"></div></div></div></div></article>`);
    });
  } catch (error) {
    status.textContent = error.message;
    results.innerHTML = `<div class="empty-state"><div>!</div><p>${error.message}</p></div>`;
  }
});
