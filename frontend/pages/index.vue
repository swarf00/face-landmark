<template>
  <div class="container">
    <img v-if="markedImage" :src="`http://localhost:8000${markedImage}`" alt="landmark">
    <p>Detect landmark of face</p>
    <el-upload
      class="upload-demo"
      ref="upload"
      name="image"
      list-type="picture"
      :file-list="fileList"
      accept=".jpg, .jpeg, .png"
      action="http://localhost:8000/api/detect/face/landmark/"
      :on-success="onSuccess"
      :on-error="onError"
      :auto-upload="false">
      <el-button slot="trigger" size="small" type="primary">select image file</el-button>
      <el-button style="margin-left: 10px;" size="small" type="success" @click="submitUpload">upload to server</el-button>
      <div class="el-upload__tip" slot="tip">jpg/png</div>
    </el-upload>
  </div>
</template>

<script>

export default {
  name: 'Index',
  data () {
    return {
      markedImage: null,
      dialogImageUrl: '',
      fileList: [],
      dialogVisible: false
    }
  },
  methods: {
    submitUpload () {
      this.$refs.upload.submit()
    },
    onSuccess (data) {
      console.log(data)
      this.markedImage = data.img_url
      this.fileList = []
    },
    onError (data) {
      this.$message.error(data)
    }
  }
}
</script>

<style lang="scss" scoped>
html, body {
  padding: 0;
  margin: 0;
  color: white;
}
ul, ol {
  list-style: none;
}
a {
  text-decoration: none;
  &:hover {
    text-decoration: none;
  }
}
input[type=file] {
  display: none;
}
p {
  margin: 30px 0;
}

.container {
  margin: 0 auto;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}

</style>
