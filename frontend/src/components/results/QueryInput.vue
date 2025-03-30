<template>
  <div class="query-input">
    <el-input
      v-model="inputValue"
      type="textarea"
      :rows="3"
      placeholder="Enter your question here..."
      class="float-animation"
    >
      <template #prefix>
        <el-icon class="input-icon"><Search /></el-icon>
      </template>
    </el-input>
  </div>
</template>

<script>
import { Search } from '@element-plus/icons-vue'

export default {
  name: 'QueryInput',
  components: {
    Search
  },
  data() {
    return {
      inputValue: ''
    }
  },
  watch: {
    inputValue(val) {
      this.$emit('update:question', val)
    }
  },
  props: {
    question: {
      type: String,
      default: ''
    }
  }
}
</script>

<style scoped>
.query-input {
  margin-bottom: 24px;
}

.input-icon {
  color: var(--primary);
}
</style>
