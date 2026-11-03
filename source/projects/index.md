---
title: 项目专区
date: 2026-11-02 23:10:00
type: page
top_img: false
aside: false
comments: false
description: 一些简单、完整、可以直接玩的小项目：番茄钟、待办清单、计算器、密码生成器、贪吃蛇、打字速度测试。全部纯前端，0 依赖。
---

<style>
  .pz { max-width: 1040px; margin: 0 auto; }
  .pz * { box-sizing: border-box; }
  .pz-intro {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: 34px 32px;
    color: #eef2ff;
    background:
      radial-gradient(circle at 88% 18%, rgba(61, 220, 151, .22), transparent 18rem),
      radial-gradient(circle at 8% 0%, rgba(108, 140, 255, .35), transparent 22rem),
      linear-gradient(135deg, #101a36, #0b1226 60%, #111c3d);
    border: 1px solid rgba(108, 140, 255, .25);
    box-shadow: 0 24px 60px rgba(15, 23, 42, .18);
  }
  .pz-kicker {
    font-size: 11px;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: #9db2ff;
    margin-bottom: 12px;
  }
  .pz-intro h2 {
    margin: 0 0 10px;
    font-size: clamp(24px, 4vw, 34px);
    line-height: 1.25;
    color: #fff;
    font-weight: 650;
  }
  .pz-intro p {
    margin: 0;
    max-width: 640px;
    color: #b9c4e6;
    font-size: 15px;
    line-height: 1.9;
  }
  .pz-stats { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
  .pz-stats span {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 13px;
    border-radius: 999px;
    border: 1px solid rgba(185, 196, 230, .2);
    background: rgba(255, 255, 255, .06);
    color: #dbe3ff;
    font-size: 12px;
  }
  .pz-stats b { color: #3ddc97; font-weight: 600; }
  .pz-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    flex-wrap: wrap;
    margin: 28px 0 18px;
  }
  .pz-filters { display: flex; gap: 7px; flex-wrap: wrap; }
  .pz-filter {
    border: 1px solid #e2e8f4;
    background: #fff;
    color: #64748b;
    border-radius: 999px;
    padding: 8px 15px;
    font-size: 13px;
    cursor: pointer;
    font-family: inherit;
    transition: .18s ease;
  }
  .pz-filter:hover { border-color: #a9bcff; color: #4056b8; }
  .pz-filter.active {
    background: #1f2a4d;
    border-color: #1f2a4d;
    color: #fff;
    box-shadow: 0 8px 20px rgba(31, 42, 77, .2);
  }
  .pz-hint { color: #94a3b8; font-size: 12px; }
  .pz-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
  }
  .pz-card {
    display: flex;
    gap: 16px;
    padding: 22px;
    border-radius: 18px;
    background: #fff;
    border: 1px solid #e6ebf5;
    box-shadow: 0 10px 30px rgba(30, 41, 59, .06);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
  }
  .pz-card:hover {
    transform: translateY(-4px);
    border-color: #b9c8ff;
    box-shadow: 0 18px 44px rgba(30, 41, 59, .12);
  }
  .pz-card[hidden] { display: none !important; }
  .pz-icon {
    flex: none;
    width: 52px;
    height: 52px;
    display: grid;
    place-items: center;
    font-size: 26px;
    border-radius: 15px;
    background: linear-gradient(135deg, #eef3ff, #e8fff6);
    border: 1px solid #e3eaff;
  }
  .pz-body { min-width: 0; display: flex; flex-direction: column; }
  .pz-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 6px;
  }
  .pz-cat {
    font-size: 10px;
    letter-spacing: .08em;
    color: #4f63c9;
    background: #eef3ff;
    padding: 3px 8px;
    border-radius: 999px;
  }
  .pz-tech { font-size: 10px; color: #94a3b8; letter-spacing: .04em; }
  .pz-body h3 {
    margin: 0 0 6px;
    font-size: 18px;
    color: #1e293b;
    font-weight: 650;
    line-height: 1.35;
  }
  .pz-body p {
    margin: 0 0 14px;
    color: #64748b;
    font-size: 13.5px;
    line-height: 1.75;
    flex: 1;
  }
  .pz-actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .pz-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    padding: 8px 13px;
    border-radius: 10px;
    border: 1px solid #dbe3f3;
    background: #fff;
    color: #475569;
    font-size: 12.5px;
    text-decoration: none;
    transition: .16s ease;
    white-space: nowrap;
  }
  .pz-btn:hover { border-color: #a9bcff; color: #4056b8; background: #f8faff; }
  .pz-btn.primary {
    background: linear-gradient(135deg, #6c8cff, #8b6cff);
    border-color: transparent;
    color: #fff;
    box-shadow: 0 8px 20px rgba(108, 140, 255, .22);
  }
  .pz-btn.primary:hover { background: linear-gradient(135deg, #7d9aff, #9b7dff); color: #fff; }
  .pz-note {
    margin-top: 24px;
    padding: 18px 20px;
    border-radius: 16px;
    border: 1px dashed #d8e0f0;
    background: #fbfcff;
    color: #7c8aa5;
    font-size: 13px;
    line-height: 1.85;
  }
  .pz-note b { color: #4056b8; }
  [data-theme="dark"] .pz-filter { background: #1b2236; border-color: #2b3450; color: #aab6d3; }
  [data-theme="dark"] .pz-filter.active { background: #e8edff; border-color: #e8edff; color: #1b2236; }
  [data-theme="dark"] .pz-card { background: #161d30; border-color: #273150; }
  [data-theme="dark"] .pz-body h3 { color: #eef2ff; }
  [data-theme="dark"] .pz-body p { color: #9aa7c7; }
  [data-theme="dark"] .pz-icon { background: #1d2742; border-color: #2b3658; }
  [data-theme="dark"] .pz-btn { background: #1b2236; border-color: #2b3450; color: #b9c4e6; }
  [data-theme="dark"] .pz-note { background: #121a2e; border-color: #273150; color: #8f9dbd; }
  @media (max-width: 760px) {
    .pz-grid { grid-template-columns: 1fr; }
    .pz-intro { padding: 26px 22px; }
    .pz-card { padding: 18px; }
    .pz-toolbar { align-items: flex-start; }
  }
</style>

<div class="pz">
  <div class="pz-intro">
    <div class="pz-kicker">Projects · 动手实验室</div>
    <h2>写点小东西，比看十篇教程更有用。</h2>
    <p>
      这里放一些简单、完整、可以直接玩的小项目。全部是纯前端（HTML + CSS + JavaScript），
      没有后端、没有构建、没有第三方依赖。点开就能用，下载源码双击就能跑。
    </p>
    <div class="pz-stats">
      <span><b>6</b> 个项目</span>
      <span><b>0</b> 依赖</span>
      <span><b>100%</b> 纯前端</span>
      <span>源码可直接下载</span>
    </div>
  </div>
  <div class="pz-toolbar">
    <div class="pz-filters" role="group" aria-label="项目分类">
      <button type="button" class="pz-filter active" data-cat="all">全部</button>
      <button type="button" class="pz-filter" data-cat="tool">效率与工具</button>
      <button type="button" class="pz-filter" data-cat="game">小游戏</button>
    </div>
    <span class="pz-hint">数据只保存在你的浏览器里</span>
  </div>
  <div class="pz-grid">
    <article class="pz-card" data-cat="tool">
      <div class="pz-icon">🍅</div>
      <div class="pz-body">
        <div class="pz-meta"><span class="pz-cat">效率工具</span><span class="pz-tech">HTML · CSS · JS</span></div>
        <h3>番茄钟</h3>
        <p>25 分钟专注 + 5 分钟休息，每 4 个番茄进入长休息。到点提示音，自动记录今日番茄和专注分钟。</p>
        <div class="pz-actions">
          <a class="pz-btn primary" href="/demos/pomodoro/">在线体验</a>
          <a class="pz-btn" href="/demos/pomodoro/index.html" download="pomodoro.html">下载源码</a>
        </div>
      </div>
    </article>
    <article class="pz-card" data-cat="tool">
      <div class="pz-icon">✅</div>
      <div class="pz-body">
        <div class="pz-meta"><span class="pz-cat">效率工具</span><span class="pz-tech">localStorage</span></div>
        <h3>待办清单</h3>
        <p>添加、勾选、双击编辑、删除，三档筛选，清除已完成。数据存在浏览器本地，刷新不丢。</p>
        <div class="pz-actions">
          <a class="pz-btn primary" href="/demos/todo/">在线体验</a>
          <a class="pz-btn" href="/demos/todo/index.html" download="todo.html">下载源码</a>
        </div>
      </div>
    </article>
    <article class="pz-card" data-cat="tool">
      <div class="pz-icon">🧮</div>
      <div class="pz-body">
        <div class="pz-meta"><span class="pz-cat">实用工具</span><span class="pz-tech">Parser · 键盘操作</span></div>
        <h3>计算器</h3>
        <p>四则运算、括号、百分号、正负号，支持键盘输入。表达式用递归下降解析，不靠 eval，历史记录可回填。</p>
        <div class="pz-actions">
          <a class="pz-btn primary" href="/demos/calculator/">在线体验</a>
          <a class="pz-btn" href="/demos/calculator/index.html" download="calculator.html">下载源码</a>
        </div>
      </div>
    </article>
    <article class="pz-card" data-cat="tool">
      <div class="pz-icon">🔐</div>
      <div class="pz-body">
        <div class="pz-meta"><span class="pz-cat">实用工具</span><span class="pz-tech">Web Crypto</span></div>
        <h3>密码生成器</h3>
        <p>长度 8–32，大小写、数字、符号自由组合，可排除易混淆字符。实时显示字符池、熵和强度，一键复制。</p>
        <div class="pz-actions">
          <a class="pz-btn primary" href="/demos/password/">在线体验</a>
          <a class="pz-btn" href="/demos/password/index.html" download="password.html">下载源码</a>
        </div>
      </div>
    </article>
    <article class="pz-card" data-cat="game">
      <div class="pz-icon">🐍</div>
      <div class="pz-body">
        <div class="pz-meta"><span class="pz-cat">小游戏</span><span class="pz-tech">Canvas · 触屏</span></div>
        <h3>贪吃蛇</h3>
        <p>Canvas 手绘网格与蛇身，方向键 / WASD / 滑动控制。吃到食物会加速，最高分自动保存在本地。</p>
        <div class="pz-actions">
          <a class="pz-btn primary" href="/demos/snake/">在线体验</a>
          <a class="pz-btn" href="/demos/snake/index.html" download="snake.html">下载源码</a>
        </div>
      </div>
    </article>
    <article class="pz-card" data-cat="game">
      <div class="pz-icon">⌨️</div>
      <div class="pz-body">
        <div class="pz-meta"><span class="pz-cat">小游戏</span><span class="pz-tech">WPM · 逐字高亮</span></div>
        <h3>打字速度测试</h3>
        <p>30 秒或 60 秒计时，实时显示 WPM、准确率和逐字对错，打完自动结算，并记录个人最佳成绩。</p>
        <div class="pz-actions">
          <a class="pz-btn primary" href="/demos/typing/">在线体验</a>
          <a class="pz-btn" href="/demos/typing/index.html" download="typing.html">下载源码</a>
        </div>
      </div>
    </article>
  </div>
  <div class="pz-note">
    <b>怎么玩：</b>点“在线体验”直接使用；点“下载源码”会得到一个独立的 HTML 文件，双击就能在浏览器里运行。
    所有项目都没有后端和外部依赖，代码也不长，适合拿来改着玩——改颜色、加功能、拆成自己的版本，都是很好的练习。
  </div>
</div>
<script>
(() => {
  const filters = [...document.querySelectorAll('.pz-filter')];
  const cards = [...document.querySelectorAll('.pz-card')];
  filters.forEach((filter) => {
    filter.addEventListener('click', () => {
      filters.forEach((item) => item.classList.toggle('active', item === filter));
      const category = filter.dataset.cat;
      cards.forEach((card) => {
        card.hidden = category !== 'all' && card.dataset.cat !== category;
      });
    });
  });
})();
</script>
