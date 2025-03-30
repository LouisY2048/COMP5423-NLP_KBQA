<template>
  <div class="method-selector">
    <h3>Retrieval Method</h3>
    <el-radio-group v-model="selectedMethod" @change="emitChange" class="method-options">
      <el-radio label="keyword" border class="method-option">
        <div class="method-option-content">
          <el-icon><Search /></el-icon>
          <div class="method-text">
            <div class="method-name">Keyword Retrieval</div>
          </div>
          <el-badge v-if="selectedMethod === 'keyword'" type="success" value="Selected" />
        </div>
      </el-radio>
      
      <el-radio label="dense" border class="method-option">
        <div class="method-option-content">
          <el-icon><HomeFilled /></el-icon>
          <div class="method-text">
            <div class="method-name">Dense Retrieval</div>          </div>
          <el-badge v-if="selectedMethod === 'dense'" type="success" value="Selected" />
        </div>
      </el-radio>
    </el-radio-group>
  </div>
</template>

<script>
import { Search, HomeFilled } from '@element-plus/icons-vue'

export default {
  name: 'MethodSelector',
  components: {
    Search,
    HomeFilled
  },
  data() {
    return {
      selectedMethod: this.method || 'keyword'
    }
  },
  props: {
    method: {
      type: String,
      default: 'keyword'
    }
  },
  methods: {
    emitChange() {
      this.$emit('update:method', this.selectedMethod)
    }
  }
}
</script>

<style scoped>
.method-selector {
  margin-bottom: 24px;
}

.method-options {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  justify-content: center;
}

.method-option {
  flex: 1;
  min-width: 250px;
  max-width: 400px;
  transition: var(--transition-smooth);
}

.method-option:hover {
  transform: scale(1.03);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.method-option-content {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
}

.method-text {
  flex: 1;
}

.method-name {
  font-weight: 600;
  color: var(--primary);
  font-size: 16px;
}

.method-description {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

@media (max-width: 768px) {
  .method-options {
    flex-direction: column;
  }
  
  .method-option {
    max-width: 100%;
  }
}
</style>
