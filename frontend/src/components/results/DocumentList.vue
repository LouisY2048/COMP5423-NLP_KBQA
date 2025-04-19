<template>
  <transition name="fade">
    <div v-if="documents && documents.length" class="document-list">
      <h3 class="result-title">
        <el-icon><Document /></el-icon>
        Related Documents
      </h3>
      <el-collapse v-model="activeDocuments" class="document-collapse">
        <el-collapse-item 
          v-for="(doc, index) in displayedDocuments" 
          :key="index"
          :title="`Document ${index + 1}`"
          :name="index"
        >
          <div class="document-content">{{ doc }}</div>
        </el-collapse-item>
      </el-collapse>
    </div>
  </transition>
</template>

<script>
import { Document } from '@element-plus/icons-vue'

export default {
  name: 'DocumentList',
  components: {
    Document
  },
  data() {
    return {
      activeDocuments: [0]  // First document open by default
    }
  },
  props: {
    documents: {
      type: Array,
      default: () => []
    }
  },
  computed: {
    displayedDocuments() {
      return this.documents.slice(0, 5)
    }
  }
}
</script>

<style scoped>
.document-list {
  margin-top: 12px;
}

.result-title {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--secondary);
}

.document-collapse {
  margin-top: 12px;
  border-radius: 8px;
  overflow: hidden;
}

.document-content {
  white-space: pre-line;
  line-height: 1.5;
  background: rgba(245, 247, 250, 0.7);
  padding: 16px;
  border-radius: 8px;
  font-size: 14px;
}

.el-collapse-item :deep(.el-collapse-item__header) {
  background: rgba(255, 255, 255, 0.6);
  padding: 0 16px;
  transition: var(--transition-smooth);
}

.el-collapse-item :deep(.el-collapse-item__header:hover) {
  background: rgba(67, 97, 238, 0.1);
  color: var(--primary);
}
</style>
