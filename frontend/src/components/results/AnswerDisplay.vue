<template>
  <transition name="fade">
    <div v-if="answer" class="answer-display">
      <h3 class="result-title">
        <el-icon><ChatDotRound /></el-icon>
        Answer
      </h3>
      <el-card class="answer-card">
        <div class="answer-content">{{ answer }}</div>
      </el-card>
    </div>
  </transition>
</template>

<script>
import { ChatDotRound } from '@element-plus/icons-vue'

export default {
  name: 'AnswerDisplay',
  components: {
    ChatDotRound
  },
  props: {
    answer: {
      type: String,
      default: ''
    }
  }
}
</script>

<style scoped>
.answer-display {
  margin-bottom: 24px;
}

.result-title {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--secondary);
}

.answer-card {
  margin-top: 12px;
  background: rgba(255, 255, 255, 0.8);
}

.answer-content {
  white-space: pre-line;
  line-height: 1.6;
  color: #333;
  font-size: 16px;
  padding: 8px;
}
</style>
