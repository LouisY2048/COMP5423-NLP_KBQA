<template>
  <div class="query-input">
    <div class="input-container">
      <el-input
        v-model="inputValue"
        type="text"
        placeholder="Enter your question here..."
        class="float-animation"
      >
        <template #prefix>
          <el-icon class="input-icon"><Search /></el-icon>
        </template>
        <template #suffix>
          <el-dropdown trigger="click" @command="handleQuestionSelect">
            <el-button type="text" class="dropdown-button">
              <el-icon><ArrowDown /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item 
                  v-for="(question, index) in presetQuestions" 
                  :key="index"
                  :command="question"
                >
                  {{ question }}
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </template>
      </el-input>
    </div>
  </div>
</template>

<script>
import { Search, ArrowDown } from '@element-plus/icons-vue'

export default {
  name: 'QueryInput',
  components: {
    Search,
    ArrowDown
  },
  data() {
    return {
      inputValue: '',
      presetQuestions: [
        'What is the capital of France',
        'who played the king in the kings speech',
        'when did the 1st world war officially end',
        'who is the director of the cia today',
        'what age can you get a tattoo in louisiana',
        'who sang the song please come to boston',
        'who was the original actor in walking tall',
        'what year did beyonce do the super bowl'
      ]
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
  },
  methods: {
    handleQuestionSelect(question) {
      this.inputValue = question
      this.$nextTick(() => {
        this.$emit('update:question', question)
        this.$emit('search', question)
      })
    }
  }
}
</script>

<style scoped>
.query-input {
  margin-bottom: 24px;
}

.input-container {
  display: flex;
  align-items: center;
}

.input-icon {
  color: var(--primary);
}

.dropdown-button {
  padding: 0;
  margin: 0;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.dropdown-button .el-icon {
  font-size: 16px;
  color: var(--el-text-color-regular);
}

:deep(.el-dropdown-menu) {
  max-width: 400px;
}

:deep(.el-dropdown-menu__item) {
  white-space: normal;
  padding: 8px 16px;
  line-height: 1.5;
}

:deep(.el-input__wrapper) {
  padding-right: 8px;
}
</style>
