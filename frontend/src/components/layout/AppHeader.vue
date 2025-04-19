<template>
  <el-header class="app-header">
    <div class="header-content">
      <h1 class="system-title">
        <span class="title-text" @mouseover="isHovered = true" @mouseleave="isHovered = false">
          <span class="text-container">
            <span class="text-primary" :class="{ 'fade-out': isHovered }">Knowledge Base Question Answering System</span>
            <span class="text-secondary" :class="{ 'fade-in': isHovered }">Answer What You Don't Know</span>
          </span>
        </span>
      </h1>
    </div>
  </el-header>
</template>

<script>
export default {
  name: 'AppHeader',
  data() {
    return {
      isHovered: false
    }
  }
}
</script>

<style scoped>
.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 16px 0;
  text-align: center;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  position: relative;
  overflow: hidden;
}

.app-header::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
  animation: shimmer 3s infinite;
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: center;
  align-items: center;
}

.system-title {
  color: white;
  margin: 0;
  font-weight: 600;
  font-size: 24px;
  letter-spacing: 0.5px;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  transition: transform 0.3s ease;
  position: relative;
  z-index: 2;
  text-align: center;
  width: 100%;
}

.system-title:hover {
  transform: scale(1.02);
}

.title-text {
  position: relative;
  display: inline-block;
  cursor: pointer;
  height: 1.2em;
  text-align: center;
}

.text-container {
  position: relative;
  display: inline-block;
  height: 100%;
  text-align: center;
}

.text-primary {
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  width: auto;
  transition: all 0.5s ease;
  opacity: 1;
  white-space: nowrap;
  background: linear-gradient(to bottom, #FDB931 0%, #FDB931 50%, #9F7928 51%, #9F7928 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  font-weight: 700;
}

.text-secondary {
  opacity: 0;
  transform: translate(-50%, 20px);
  background: linear-gradient(45deg, #00ff00, #00bfff, #1e90ff);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.fade-out {
  opacity: 0;
  transform: translate(-50%, -20px);
}

.fade-in {
  opacity: 1;
  transform: translate(-50%, 0);
}

.title-text::after {
  content: '';
  position: absolute;
  width: 100%;
  height: 2px;
  bottom: -4px;
  left: 0;
  background-color: white;
  transform: scaleX(0);
  transform-origin: right;
  transition: transform 0.3s ease;
}

.title-text:hover::after {
  transform: scaleX(1);
  transform-origin: left;
}
</style>
