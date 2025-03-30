<template>
  <el-header class="app-header">
    <div class="header-content">
      <h1 class="system-title">Knowledge Base Question Answering System</h1>
    </div>
  </el-header>
</template>

<script>
export default {
  name: 'AppHeader'
}
</script>

<style scoped>
.app-header {
  background-color: var(--primary);
  padding: 12px 0;
  text-align: center;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
}

.system-title {
  color: white;
  margin: 0;
  font-weight: 600;
  font-size: 24px;
  letter-spacing: 0.5px;
  text-shadow: var(--text-shadow);
}
</style>
