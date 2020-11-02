#[cfg(feature = "shaderc")]
pub struct GLSLCompile {
    code: String,
    compiler: shaderc::Compiler,
}

#[cfg(feature = "shaderc")]
impl GLSLCompile {
    pub fn new(code: &str) -> Self {
        GLSLCompile {
            code: code.to_string(),
            compiler: shaderc::Compiler::new().unwrap(),
        }
    }
    pub fn compile(&mut self, entry: &str) -> Result<Vec<u32>, ()> {
        let bin = self
            .compiler
            .compile_into_spirv(
                &self.code,
                shaderc::ShaderKind::Compute,
                "name",
                entry,
                None,
            )
            .unwrap();
        Ok(bin.as_binary().to_vec())
    }
}
